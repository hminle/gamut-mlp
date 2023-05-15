import colour

ILLUMINANT_D50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
ILLUMINANT_D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
prophotorgb_colourspace = colour.models.RGB_COLOURSPACE_PROPHOTO_RGB.chromatically_adapt(
    ILLUMINANT_D65
)
srgb_colourspace = colour.models.RGB_COLOURSPACE_sRGB
