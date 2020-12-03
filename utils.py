import re
import torch

# load pretrained weights from tf to pytorch model
# adapted from https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
def load_weights(model, weight_dict):
    for name, array in weight_dict:
        name = name.split('/')[-1]
        name = name.split('.')
        if name[0] in ['Adam', 'Adam_1'] :
            continue 
        pointer = model
        transpose_axes = None

        # print("initialize PyTorch weight {}".format(name))        
        # iterate along the scopes and move pointer accordingly
        for m_name in name[1:]:
            # handle digits in module list
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]

            # Convert parameters final names to pytorch equivalents
            if l[0] == 'Filters' or l[0] == 'W':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'Biases' or l[0] == 'b':
                pointer = getattr(pointer, 'bias')
            
            elif l[0] == 'scale':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'offset':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'moving_mean':
                pointer = getattr(pointer, 'running_mean')
            elif l[0] == 'moving_variance':
                pointer = getattr(pointer, 'running_var')
            
            else:
                pointer = getattr(pointer, l[0])

            # sometimes tf and pytorch conventions are transposed
            if l[0] == 'W':
                transpose_axes = (1, 0)
            if l[0] == 'Filters':
                transpose_axes = (3, 2, 0, 1)

            # If module list, access the sub-module with the right number
            if len(l) >= 2:
                num = int(l[1])-1 # our torch indexes start at 0, but TF names start at 1
                pointer = pointer[num]
                
        if transpose_axes:
            array = array.transpose(transpose_axes)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pointer.data = torch.from_numpy(array)


def save_weights(model, save_path):
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict, save_path)
