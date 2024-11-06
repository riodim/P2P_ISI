from torchviz import make_dot
import torch
from model.model_loss import OTAPulse
import config

def visualize_model():
    # Initialize the model based on config parameters
    model = OTAPulse(input_size=config.input_size, output_size=config.num_points, hidden_layers=config.hidden_layers).to(config.device)

    # Generate a sample input tensor
    sample_input = torch.randn(1, config.input_size).to(config.device)
    output = model(sample_input)

    # Generate the model graph with torchviz
    model_viz = make_dot(output, params=dict(model.named_parameters()))

    # Customize nodes with descriptive labels
    layer_counter = 1
    for layer_name, layer in model.named_modules():
        if layer_name:  # Avoid the root node
            layer_type = type(layer).__name__
            if isinstance(layer, torch.nn.Linear):
                layer_details = f"Layer {layer_counter}: {layer_type}\n"
                layer_details += f"Input: {layer.in_features}, Output: {layer.out_features}\n"
                model_viz.node(layer_name, layer_details, style='filled', color='lightblue')
                layer_counter += 1
            elif isinstance(layer, torch.nn.ReLU):
                model_viz.node(layer_name, f"Activation: {layer_type}", style='filled', color='lightgreen')

    # Save the visualized model structure
    model_viz.format = "png"
    model_viz.render("OTAPulse_model_structure_custom")
    print("Model structure saved as OTAPulse_model_structure_custom_2.png")

# Call the visualization function
visualize_model()
