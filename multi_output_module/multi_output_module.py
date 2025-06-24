import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

class Multi_output_module(nn.Module):
    def __init__(self, num_heads, base_model, device):
        super(Multi_output_module, self).__init__()

        self.device = device
        self.num_heads = num_heads
        self.activation = nn.ReLU()
        self.base_model = base_model.to(self.device)
        self.droprate = 0.3

        self.penultimate_output = None
        self._register_penultimate_hook()

        self.last_layer = list(self.base_model.children())[-1]

        if isinstance(self.last_layer, nn.Sequential): 
            self.last_layer_input_dim = self.last_layer[0].in_features
            self.num_classes = self.last_layer[-1].out_features
        else:
            self.last_layer_input_dim = self.last_layer.in_features
            self.num_classes = self.last_layer.out_features

        self.input_heads = nn.ModuleList([
            self.initialize_head(copy.deepcopy(self.last_layer))
            for _ in range(self.num_heads)
        ]).to(self.device)

        input_dim = self.num_classes * self.num_heads
        self.shared_layers = nn.Linear(input_dim, input_dim).to(self.device)

        self.output_layers = nn.ModuleList([
            nn.Linear(input_dim, self.num_classes)
            for _ in range(self.num_heads)
            ]).to(self.device)

    def _register_penultimate_hook(self):

        def hook(module, input, output):
            self.penultimate_output = input[0].detach()
        last_layer = list(self.base_model.children())[-1]
        last_layer.register_forward_hook(hook)

    def initialize_head(self, layer):

        def initialize_layer(module):
            if isinstance(module, nn.Linear):
                # Initialize Linear layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax, nn.Dropout, nn.Flatten)):
                pass
            else:
                raise ValueError(f"Unsupported module type in the last layer: {type(module)}")

        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                initialize_layer(sub_layer)
        else:
            initialize_layer(layer)
        
        return layer

    def forward(self, x, train_infer):
        self.base_model.eval()

        with torch.no_grad():
            x = x.to(self.device)
            if train_infer == 'training':
                x = self.base_model(x)
                x = self.activation(self.penultimate_output).view(-1, self.num_heads, self.last_layer_input_dim)

            elif train_infer == 'inference':
                out_base = self.base_model(x)
                x = self.activation(self.penultimate_output).unsqueeze(1).repeat(1, self.num_heads, 1)

            else:
                raise ValueError('train_infer must be either "training" or "inference"')
        
        processed_heads = []
        for i in range(self.num_heads):
            x_i = x[:, i, :]
            x_i = self.activation(self.input_heads[i](x_i))
            processed_heads.append(x_i)
        
        combined_processed_heads = torch.cat(processed_heads, dim=1)
        x = F.dropout(self.activation(self.shared_layers(combined_processed_heads)), p=self.droprate, training=self.training)

        output_heads = []
        for i in range(self.num_heads):
            output_head_i = self.output_layers[i](x)
            output_heads.append(output_head_i)

        if train_infer == 'training':
            x = torch.cat(output_heads, dim=1)
            x = x.view(-1, self.num_heads, self.num_classes)

        elif train_infer == 'inference':
            output_heads.append(out_base) # Add the last layer of the model too if it is inference
            x = torch.cat(output_heads, dim=1) 
            x = x.view(-1, self.num_heads+1, self.num_classes)

        return x