import torch


# input, kernel, output are tensors on the GPU
def conv1d(
    input: torch.Tensor,
    kernel: torch.Tensor,  # out, in, kl
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    input_unfold = input.unfold(-1, kernel_size, 1)  # bs, in, l, kl
    input_unfold = input_unfold.transpose(-2, -3)
    input_unfold = input_unfold.reshape(
        [input_unfold.shape[0], input_unfold.shape[1], -1]
    )
    kernel = kernel.reshape([kernel.shape[0], -1])
    output.data = (input_unfold @ kernel.transpose(0, -1)).transpose(-1, -2)


def my_unfold(input: torch.Tensor, k_h, k_w) -> torch.Tensor:
    bs, in_c, in_h, in_w = input.shape
    o_h, o_w = in_h - k_h + 1, in_w - k_w + 1
    # bs, in_c, in_h, in_w -> bs, in_c, o_h, in_w, k_h
    input = input.unfold(2, k_h, 1)
    # bs, in_c, o_h, in_w, k_h -> bs, in_c, o_h, o_w, k_h, k_w
    input = input.unfold(3, k_w, 1)

    out = input.permute(0, 1, 4, 5, 2, 3)
    out = out.reshape(bs, in_c * k_h * k_w, o_h * o_w)
    return out


def conv2d(
    input: torch.Tensor,  # bs, in, h, w
    kernel: torch.Tensor,  # out, in, kh, kw
    output: torch.Tensor,  # bs, out, h, w
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    bs, in_status, _, _ = input.shape
    out_status = kernel.shape[0]
    oh, ow = input_rows - kernel_rows + 1, input_cols - kernel_cols + 1
    input = my_unfold(input, kernel_rows, kernel_cols).transpose(1, 2)
    kernel = kernel.reshape(kernel.shape[0], -1).transpose(0, 1)  # (in, kh, kw), out
    output = input @ kernel  # bs, o_h * o_w, k_h * k_w*in_c @
    output = output.reshape(bs, out_status, oh, ow)
    print(output)


# input, kernel, output are tensors on the GPU
def conv3d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
):
    bs, in_c, in_h, in_w, in_d = input.shape
    out_c, in_c, k_h, k_w, k_d = kernel.shape
    out_h, out_w, out_d = in_h - k_h + 1, in_w - k_w + 1, in_d - k_d + 1
    # bs, in_c, in_h, in_w, in_d -> bs, (in_c, in_h), k_w*k_d,o_w*o_d
    input = input.reshape(bs, in_c * in_h, in_w, in_d)
    input = torch.nn.functional.unfold(input=input, kernel_size=(k_w, k_d)).reshape(
        bs, in_c, in_h, k_w * k_d, out_w * out_d
    )
    # bs, in_c, in_h, k_w * k_d, out_w * out_d -> bs, in_c, out_h, k_w * k_d, out_w * out_d, k_h
    input = input.unfold(2, k_h, 1)
    # bs, in_c, out_h, k_w * k_d, out_w * out_d, k_h -> bs, in_c, out_h, k_w * k_d, k_h, out_w * out_d
    input = input.transpose(-1, -2)
    # bs, in_c, out_h, k_w * k_d, k_h, out_w * out_d -> bs, in_c, k_w * k_d, k_h, out_w * out_d, out_h
    input = input.transpose(2, 4)
    input = input.reshape(bs, in_c * k_w * k_d * k_h, out_w * out_d * out_h)
    kernel = kernel.reshape(out_c, -1).transpose(0, 1)  # in_c * k_w * k_d * k_h, out_c
    output_t = (input.transpose(1, 2)) @ kernel  # bs, o, out_c
    output_t = output_t.transpose(1, 2)
    output_t = output_t.reshape(bs, out_c, out_h, out_w, out_d)
    output.data = output_t
    print(output)


def ref_conv(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
):
    output.data = torch.nn.functional.conv3d(
        input=input,
        weight=kernel,
    )
    print(output)
    return output


input_tensor = torch.randn([1, 1, 4, 4, 4], dtype=torch.float32)  # bs, in, sl
kernel_tensor = torch.Tensor([[[[[1, 1], [1, 1], [0, 1]]]]])  # out, in, kl
output_tensor1 = torch.zeros([1, 1, 3, 3, 3], dtype=torch.float32)  # bs,out,sl_new
output_tensor2 = torch.zeros([1, 1, 3, 3, 3], dtype=torch.float32)
ref_conv(input_tensor, kernel_tensor, output_tensor1)
conv3d(input_tensor, kernel_tensor, output_tensor2)
