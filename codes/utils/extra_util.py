import cv2
import numpy as np
import torch.nn as nn
import torch
import os
import colour
import math
import csv

def get_HW_boundary(patch_boundary, h, w, pH, sH, pW, sW):
    H_low_ind = max(pH * sH - patch_boundary, 0)
    H_high_ind = min((pH + 1) * sH + patch_boundary, h)
    W_low_ind = max(pW * sW - patch_boundary, 0)
    W_high_ind = min((pW + 1) * sW + patch_boundary, w)
    return H_low_ind, H_high_ind, W_low_ind, W_high_ind

def psnrEvaluation(dict_evaluationUnit=None, path_csv_psnr=None, name_pd_target=None, height_start=None, height_end=None, width_start=None, width_end=None):
	if not os.path.exists(path_csv_psnr):
		with open(path_csv_psnr, 'a') as f:
			csv_writer = csv.writer(f)
			header = ['id', 'height_start', 'height_end', 'width_start', 'width_end']
			for name_pd in dict_evaluationUnit['pds'].keys():
				header.append(name_pd)
			header.append('delta')
			csv_writer.writerow(header)

	list_row = [dict_evaluationUnit['id'], height_start, height_end, width_start, width_end]
	dict_psnr = {}
	for name_pd in dict_evaluationUnit['pds'].keys():
		psnr = calculate_psnr(img1=dict_evaluationUnit['gt'], img2=dict_evaluationUnit['pds'][name_pd])
		list_row.append(psnr)
		dict_psnr[name_pd] = psnr
	list_psnr_exceptTarget = []
	for key_name_pd in dict_psnr.keys():
		if key_name_pd != name_pd_target:
			list_psnr_exceptTarget.append(dict_psnr[key_name_pd])
	list_row.append(dict_psnr[name_pd_target] - max(list_psnr_exceptTarget))
	with open(path_csv_psnr, 'a') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(list_row)
	return

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32) # np.float64
    img2 = img2.astype(np.float32) # np.float64
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 20 * math.log10(1.0 / math.sqrt(mse))

def calculate_hdr_deltaITP(img1, img2):
    img1 = img1[:, :, [2, 1, 0]]
    img2 = img2[:, :, [2, 1, 0]]
    img1 = colour.models.eotf_ST2084(img1)
    img2 = colour.models.eotf_ST2084(img2)
    img1_ictcp = colour.RGB_to_ICTCP(img1)
    img2_ictcp = colour.RGB_to_ICTCP(img2)
    delta_ITP = 720 * np.sqrt((img1_ictcp[:,:,0] - img2_ictcp[:,:,0]) ** 2
                            + 0.25 * ((img1_ictcp[:,:,1] - img2_ictcp[:,:,1]) ** 2)
                            + (img1_ictcp[:,:,2] - img2_ictcp[:,:,2]) ** 2)
    return np.mean(delta_ITP)


'''
---- 1) FLOPs: floating point operations
---- 2) #Activations: the number of elements of all ‘Conv2d’ outputs
---- 3) #Conv2d: the number of ‘Conv2d’ layers
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 21/July/2020
# --------------------------------------------
# Reference
https://github.com/sovrasov/flops-counter.pytorch.git

# If you use this code, please consider the following citation:

@inproceedings{zhang2020aim, % 
  title={AIM 2020 Challenge on Efficient Super-Resolution: Methods and Results},
  author={Kai Zhang and Martin Danelljan and Yawei Li and Radu Timofte and others},
  booktitle={European Conference on Computer Vision Workshops},
  year={2020}
}
# --------------------------------------------
'''

def traverse_under_folder(folder_root='/home/ubuntu/'):
	folder_leaf = []
	folder_branch = []
	file_leaf = []
	
	index = 0
	for dirpath, subdirnames, filenames in os.walk(folder_root):
	    index += 1
	    if len(subdirnames) == 0:
	        folder_leaf.append(dirpath)
	    else:
	        folder_branch.append(dirpath)
	    for i in range(len(filenames)):
	        file_leaf.append(os.path.join(dirpath, filenames[i]))

	return folder_leaf, folder_branch, file_leaf

def get_model_flops(model, input_res, print_per_layer_stat=True,
                              input_constructor=None):
    assert type(input_res) is tuple, 'Please provide the size of the input image.'
    assert len(input_res) >= 3, 'Input image should have 3 dimensions.'
    flops_model = add_flops_counting_methods(model)
    flops_model.eval().start_flops_count()
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        device = list(flops_model.parameters())[-1].device
        batch = torch.FloatTensor(1, *input_res).to(device)
        _ = flops_model(batch)

    if print_per_layer_stat:
        print_model_with_flops(flops_model)
    flops_count = flops_model.compute_average_flops_cost()
    flops_model.stop_flops_count()

    return flops_count

def get_model_activation(model, input_res, input_constructor=None):
    assert type(input_res) is tuple, 'Please provide the size of the input image.'
    assert len(input_res) >= 3, 'Input image should have 3 dimensions.'
    activation_model = add_activation_counting_methods(model)
    activation_model.eval().start_activation_count()
    if input_constructor:
        input = input_constructor(input_res)
        _ = activation_model(**input)
    else:
        device = list(activation_model.parameters())[-1].device
        batch = torch.FloatTensor(1, *input_res).to(device)
        _ = activation_model(batch)

    activation_count, num_conv = activation_model.compute_average_activation_cost()
    activation_model.stop_activation_count()

    return activation_count, num_conv


def get_model_complexity_info(model, input_res, print_per_layer_stat=True, as_strings=True,
                              input_constructor=None):
    assert type(input_res) is tuple
    assert len(input_res) >= 3
    flops_model = add_flops_counting_methods(model)
    flops_model.eval().start_flops_count()
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        batch = torch.FloatTensor(1, *input_res).cuda()
        _ = flops_model(batch)

    if print_per_layer_stat:
        print_model_with_flops(flops_model)
    flops_count = flops_model.compute_average_flops_cost()
    params_count = get_model_parameters_number(flops_model)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num):
    if params_num // 10 ** 6 > 0:
        return str(round(params_num / 10 ** 6, 2)) + ' M'
    elif params_num // 10 ** 3:
        return str(round(params_num / 10 ** 3, 2)) + ' k'
    else:
        return str(params_num)


def print_model_with_flops(model, units='GMac', precision=3):
    total_flops = model.compute_average_flops_cost()

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_flops_cost = self.accumulate_flops()
        return ', '.join([flops_to_string(accumulated_flops_cost, units=units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    # embed()
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()
    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__

    return flops_sum


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return

        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            handle = module.register_forward_hook(conv_flops_counter_hook)
        elif isinstance(module, (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
            handle = module.register_forward_hook(relu_flops_counter_hook)
        elif isinstance(module, nn.Linear):
            handle = module.register_forward_hook(linear_flops_counter_hook)
        elif isinstance(module, (nn.BatchNorm2d)):
            handle = module.register_forward_hook(bn_flops_counter_hook)
        else:
            handle = module.register_forward_hook(empty_flops_counter_hook)
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0


# ---- Internal functions
def is_supported_instance(module):
    if isinstance(module,
                  (
                          nn.Conv2d, nn.ConvTranspose2d,
                          nn.BatchNorm2d,
                          nn.Linear,
                          nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6,
                  )):
        return True

    return False


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    # input = input[0]

    batch_size = output.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = np.prod(kernel_dims) * in_channels * filters_per_channel

    active_elements_count = batch_size * np.prod(output_dims)
    overall_conv_flops = int(conv_per_position_flops) * int(active_elements_count)

    # overall_flops = overall_conv_flops

    conv_module.__flops__ += int(overall_conv_flops)
    # conv_module.__output_dims__ = output_dims


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)
    # print(module.__flops__, id(module))
    # print(module)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    if len(input.shape) == 1:
        batch_size = 1
        module.__flops__ += int(batch_size * input.shape[0] * output.shape[0])
    else:
        batch_size = input.shape[0]
        module.__flops__ += int(batch_size * input.shape[1] * output.shape[1])


def bn_flops_counter_hook(module, input, output):
    # input = input[0]
    # TODO: need to check here
    # batch_flops = np.prod(input.shape)
    # if module.affine:
    #     batch_flops *= 2
    # module.__flops__ += int(batch_flops)
    batch = output.shape[0]
    output_dims = output.shape[2:]
    channels = module.num_features
    batch_flops = batch * channels * np.prod(output_dims)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


# ---- Count the number of convolutional layers and the activation
def add_activation_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    # embed()
    net_main_module.start_activation_count = start_activation_count.__get__(net_main_module)
    net_main_module.stop_activation_count = stop_activation_count.__get__(net_main_module)
    net_main_module.reset_activation_count = reset_activation_count.__get__(net_main_module)
    net_main_module.compute_average_activation_cost = compute_average_activation_cost.__get__(net_main_module)

    net_main_module.reset_activation_count()
    return net_main_module


def compute_average_activation_cost(self):
    """
    A method that will be available after add_activation_counting_methods() is called
    on a desired net object.

    Returns current mean activation consumption per image.

    """

    activation_sum = 0
    num_conv = 0
    for module in self.modules():
        if is_supported_instance_for_activation(module):
            print(str(type(module)) + str(module.__activation__ / (1920 * 1080)))
            activation_sum += module.__activation__
            num_conv += module.__num_conv__
    return activation_sum, num_conv


def start_activation_count(self):
    """
    A method that will be available after add_activation_counting_methods() is called
    on a desired net object.

    Activates the computation of mean activation consumption per image.
    Call it before you run the network.

    """
    self.apply(add_activation_counter_hook_function)


def stop_activation_count(self):
    """
    A method that will be available after add_activation_counting_methods() is called
    on a desired net object.

    Stops computing the mean activation consumption per image.
    Call whenever you want to pause the computation.

    """
    self.apply(remove_activation_counter_hook_function)


def reset_activation_count(self):
    """
    A method that will be available after add_activation_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    self.apply(add_activation_counter_variable_or_reset)


def add_activation_counter_hook_function(module):
    if is_supported_instance_for_activation(module):
        if hasattr(module, '__activation_handle__'):
            return

        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            handle = module.register_forward_hook(conv_activation_counter_hook)
            module.__activation_handle__ = handle


def remove_activation_counter_hook_function(module):
    if is_supported_instance_for_activation(module):
        if hasattr(module, '__activation_handle__'):
            module.__activation_handle__.remove()
            del module.__activation_handle__


def add_activation_counter_variable_or_reset(module):
    if is_supported_instance_for_activation(module):
        module.__activation__ = 0
        module.__num_conv__ = 0


def is_supported_instance_for_activation(module):
    if isinstance(module,
                  (
                          nn.Conv2d, nn.ConvTranspose2d,
                  )):
        return True

    return False

def conv_activation_counter_hook(module, input, output):
    """
    Calculate the activations in the convolutional operation.
    Reference: Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár, Designing Network Design Spaces.
    :param module:
    :param input:
    :param output:
    :return:
    """
    module.__activation__ += output.numel()
    module.__num_conv__ += 1


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


def dconv_flops_counter_hook(dconv_module, input, output):
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    m_channels, in_channels, kernel_dim1, _, = dconv_module.weight.shape
    out_channels, _, kernel_dim2, _, = dconv_module.projection.shape
    # groups = dconv_module.groups

    # filters_per_channel = out_channels // groups
    conv_per_position_flops1 = kernel_dim1 ** 2 * in_channels * m_channels
    conv_per_position_flops2 = kernel_dim2 ** 2 * out_channels * m_channels
    active_elements_count = batch_size * np.prod(output_dims)

    overall_conv_flops = (conv_per_position_flops1 + conv_per_position_flops2) * active_elements_count
    overall_flops = overall_conv_flops

    dconv_module.__flops__ += int(overall_flops)
    # dconv_module.__output_dims__ = output_dims









def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.
    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.
    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.
    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()

def calculate_ssim(img, img2, crop_border=0, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()