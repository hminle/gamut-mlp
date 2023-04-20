import torch
from torch.nn import functional as F
from tqdm import tqdm


def tinycudnn_training_loop(
    n_steps: int,
    batch_size: int,
    device: torch.device,
    inputTensor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    checkpoint_path: str = None,
    using_jit: bool = True,
) -> None:

    if using_jit:
        image = torch.jit.trace(
            inputTensor, torch.rand([batch_size], device=device, dtype=torch.float32)
        )
    else:
        image = inputTensor

    for i in range(n_steps):
        optimizer.zero_grad()

        batch = torch.rand([batch_size], device=device, dtype=torch.float32)
        inp, groundtruth = image(batch)
        output = model(inp)
        output = output + inp[:, -3:]
        loss = F.mse_loss(output, groundtruth)

        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)

