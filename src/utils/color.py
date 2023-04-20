from warnings import warn

import colour
import numpy as np
from colour.models import RGB_COLOURSPACE_PROPHOTO_RGB, RGB_COLOURSPACE_sRGB, matrix_RGB_to_RGB

ILLUMINANT_D50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
ILLUMINANT_D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]


def to_single(x):
    if x.dtype == np.uint8:
        return np.asarray(x, dtype=np.single) / 255.0  # we prefer float32
    elif x.dtype == np.uint16:
        return np.asarray(x, dtype=np.single) / 65535.0


def to_uint8(x):
    return np.asarray(x * 255, dtype=np.uint8)


def to_uint16(x):
    return np.asarray(x * 65535, dtype=np.uint16)


def is_valid_type(x):
    return x.dtype in (np.float16, np.float32, np.float64)


def is_valid_domain(x):
    return 0.0 <= np.min(x) and np.max(x) <= 1.0


def encode_srgb(x):
    assert is_valid_type(x)
    if not is_valid_domain(x):
        warn("The given input is not in the valid domain. Please check the values.")
        x = np.clip(x, 0, 1)
    return colour.cctf_encoding(x, function="sRGB")


def decode_srgb(x):
    assert is_valid_type(x) and is_valid_domain(x)

    # sRGB de-gamma (https://en.wikipedia.org/wiki/SRGB); It is faster then using colour's function.
    return np.clip(np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4), 0, 1)


def encode_prop(x):
    assert is_valid_type(x)

    if not is_valid_domain(x):
        warn("The given input is not in the valid domain. Please check the values.")
        x = np.clip(x, 0, 1)

    # colour.models.cctf_encoding_ROMMRGB/colour.models.cctf_encoding_ProPhotoRGB
    # It is faster then using colour's function.
    return np.clip(np.where(x < 0.001953125, x * 16, np.power(x, 1 / 1.8)), 0, 1)


def decode_prop(x):
    assert is_valid_type(x) and is_valid_domain(x)

    # colour.models.cctf_decoding_ROMMRGB; It is faster then using colour's function.
    return np.clip(np.where(x < 0.03125, x / 16.0, np.power(x, 1.8)), 0, 1)


def decode_displayp3(x):
    return colour.models.cctf_decoding(x, function="sRGB")


def decode_adobergb(x):
    return colour.models.cctf_decoding(x, function="Gamma 2.2")


def srgb_to_prop_cat02(x):
    #     assert is_valid_type(x) and is_valid_domain(x)
    m_srgb_to_prop = matrix_RGB_to_RGB(
        RGB_COLOURSPACE_sRGB, RGB_COLOURSPACE_PROPHOTO_RGB, chromatic_adaptation_transform="CAT02"
    )
    return np.einsum("...ij,...j->...i", m_srgb_to_prop, x)


def srgb_to_prop_bradford(x):
    assert is_valid_type(x) and is_valid_domain(x)
    m_srgb_to_prop = matrix_RGB_to_RGB(
        RGB_COLOURSPACE_sRGB,
        RGB_COLOURSPACE_PROPHOTO_RGB,
        chromatic_adaptation_transform="Bradford",
    )
    return np.einsum("...ij,...j->...i", m_srgb_to_prop, x)


def prop_to_srgb_cat02(x):
    assert is_valid_type(x) and is_valid_domain(x)
    m_prop_to_srgb = matrix_RGB_to_RGB(
        RGB_COLOURSPACE_PROPHOTO_RGB, RGB_COLOURSPACE_sRGB, chromatic_adaptation_transform="CAT02"
    )
    # If x is HxWx3
    # the below einsum is equivalent to
    # np.matmul(m_prop_to_srgb, x.reshape(-1, 3).T).T.reshape(x.shape)
    return np.einsum("...ij,...j->...i", m_prop_to_srgb, x)


def srgb_to_Lab_D50(img_srgb):
    # img_srgb has dim = MxNx3
    # img_srgb scale: 0-1 and already decoded
    # Scale:
    # L: 0-100
    # a, b: -100 - 100
    # need to normalize
    srgb_colourspace = colour.models.RGB_COLOURSPACE_sRGB.chromatically_adapt(ILLUMINANT_D50)

    XYZ = colour.models.RGB_to_XYZ(
        img_srgb.reshape(-1, 3),
        srgb_colourspace.whitepoint,
        srgb_colourspace.whitepoint,
        srgb_colourspace.matrix_RGB_to_XYZ,
    )
    Lab_srgb = colour.XYZ_to_Lab(XYZ, illuminant=ILLUMINANT_D50)

    return Lab_srgb


def lab_to_srgb_D50(Lab):
    # Scale:
    # L: 0-100
    # a, b: -100 - 100
    srgb_colourspace = colour.models.RGB_COLOURSPACE_sRGB.chromatically_adapt(ILLUMINANT_D50)
    XYZ = colour.Lab_to_XYZ(Lab, illuminant=ILLUMINANT_D50)

    img = colour.models.XYZ_to_RGB(
        XYZ,
        srgb_colourspace.whitepoint,
        srgb_colourspace.whitepoint,
        srgb_colourspace.matrix_XYZ_to_RGB,
    )
    # img has dim Nx3, and not encoded yet.
    return img


def prop_to_Lab_D50(img_prop):
    # img_srgb has dim = MxNx3
    # img_srgb scale: 0-1 and already decoded
    prophotorgb_colourspace = colour.models.RGB_COLOURSPACE_PROPHOTO_RGB

    XYZ = colour.models.RGB_to_XYZ(
        img_prop.reshape(-1, 3),
        prophotorgb_colourspace.whitepoint,
        prophotorgb_colourspace.whitepoint,
        prophotorgb_colourspace.matrix_RGB_to_XYZ,
    )
    Lab_prop = colour.XYZ_to_Lab(XYZ, illuminant=ILLUMINANT_D50)
    # Scale:
    # L: 0-100
    # a, b: -100 - 100
    # need to normalize
    return Lab_prop


def lab_to_prop_D50(Lab):
    # Scale:
    # L: 0-100
    # a, b: -100 - 100
    prophotorgb_colourspace = colour.models.RGB_COLOURSPACE_PROPHOTO_RGB
    XYZ = colour.Lab_to_XYZ(Lab, illuminant=ILLUMINANT_D50)

    img = colour.models.XYZ_to_RGB(
        XYZ,
        prophotorgb_colourspace.whitepoint,
        prophotorgb_colourspace.whitepoint,
        prophotorgb_colourspace.matrix_XYZ_to_RGB,
    )
    # img has dim Nx3, and not encoded yet.
    return img


def srgb_to_Lab(img_srgb, illuminant=ILLUMINANT_D65):
    # img_srgb has dim = MxNx3
    # img_srgb scale: 0-1 and already decoded
    # Scale:
    # L: 0-100
    # a, b: -100 - 100
    # need to normalize
    srgb_colourspace = colour.models.RGB_COLOURSPACE_sRGB

    XYZ = colour.models.RGB_to_XYZ(
        img_srgb.reshape(-1, 3),
        srgb_colourspace.whitepoint,
        srgb_colourspace.whitepoint,
        srgb_colourspace.matrix_RGB_to_XYZ,
    )
    Lab_srgb = colour.XYZ_to_Lab(XYZ, illuminant=illuminant)

    return Lab_srgb


def lab_to_srgb(Lab, illuminant=ILLUMINANT_D65):
    # Scale:
    # L: 0-100
    # a, b: -100 - 100
    srgb_colourspace = colour.models.RGB_COLOURSPACE_sRGB
    XYZ = colour.Lab_to_XYZ(Lab, illuminant=illuminant)

    img = colour.models.XYZ_to_RGB(
        XYZ,
        srgb_colourspace.whitepoint,
        srgb_colourspace.whitepoint,
        srgb_colourspace.matrix_XYZ_to_RGB,
    )
    # img has dim Nx3, and not encoded yet.
    return img


def prop_to_Lab_D65(img_prop):
    # img_srgb has dim = MxNx3
    # img_srgb scale: 0-1 and already decoded
    prophotorgb_colourspace = colour.models.RGB_COLOURSPACE_PROPHOTO_RGB.chromatically_adapt(
        ILLUMINANT_D65
    )

    XYZ = colour.models.RGB_to_XYZ(
        img_prop.reshape(-1, 3),
        prophotorgb_colourspace.whitepoint,
        prophotorgb_colourspace.whitepoint,
        prophotorgb_colourspace.matrix_RGB_to_XYZ,
    )
    Lab_prop = colour.XYZ_to_Lab(XYZ, illuminant=ILLUMINANT_D65)
    # Scale:
    # L: 0-100
    # a, b: -100 - 100
    # need to normalize
    return Lab_prop


def lab_to_prop_D65(Lab):
    # Scale:
    # L: 0-100
    # a, b: -100 - 100
    prophotorgb_colourspace = colour.models.RGB_COLOURSPACE_PROPHOTO_RGB.chromatically_adapt(
        ILLUMINANT_D65
    )
    XYZ = colour.Lab_to_XYZ(Lab, illuminant=ILLUMINANT_D65)

    img = colour.models.XYZ_to_RGB(
        XYZ,
        prophotorgb_colourspace.whitepoint,
        prophotorgb_colourspace.whitepoint,
        prophotorgb_colourspace.matrix_XYZ_to_RGB,
    )
    # img has dim Nx3, and not encoded yet.
    return img
