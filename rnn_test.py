import torch
import argparse
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
from pathlib import Path

from geometry_path import kasper_loaders, sample_images, plot_recurrence_grid_joint_embedding, plot_cache_models_joint_embedding
from analyze_representations import load_model_path

# visualizing the model 
# from tikz_visualizer import visualize_blt, TikzComputationGraphVisualizer

# better traceback hilighting 
from rich.traceback import install
install()

plt.style.use('ggplot') 
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.dpi'] = 300

def build_args():
    parser = argparse.ArgumentParser(description='Kasper Dataset Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model-path', type=str,
                    default="./blt_local_cache/imagenet_vggface2_blt_bl_top2linear_run15/blt_full_objects.pt",
                    help='path to the trained model checkpoint')
    parser.add_argument('--layer-categories', type=str,
                        default=json.dumps({
                            "Layers 1": ["conv_0_0", "conv_1_1", "conv_2_2"],
                            "Layers 2": ["conv_3_3", "conv_4_4", "conv_5_5"]
                        }),
                        help='JSON string describing the layer categories')
    args = parser.parse_args('')

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args.use_cuda = use_cuda
    args.device = device
    args.layer_categories = OrderedDict(json.loads(args.layer_categories))
    return args

def main():
    args = build_args()
    path = args.model_path
    layer_categories = args.layer_categories
    train_loader, test_loader = kasper_loaders(args)

    model_recurrent, gap, _ = load_model_path(path, print_model=True)

    # train_nodes, _ = get_graph_node_names(model_recurrent)
    # print('The computational steps in the network are: \n', train_nodes)

    # imgs, labels = sample_images(test_loader, n=20)
    # return_layers = ['conv_2', 'conv_2_1', 'conv_2_2', 'conv_2_3', 'conv_2_4', 'conv_2_5', 'conv_2_6', 'conv_2_7', 'conv_2_8', 'conv_2_9']
    # model_features = extract_features(model_recurrent, imgs.to(args.device), return_layers)

    # rdms, rdms_dict = calc_rdms(model_features)
    # plot_rdms(rdms_dict)

    imgs, labels = sample_images(test_loader, n=50)
    max_steps_to_plot = model_recurrent.times

    # plot_recurrence_grid(
    #     model_recurrent,
    #     imgs,
    #     labels,
    #     layer_categories,
    #     steps=model_recurrent.times,
    #     max_steps=max_steps_to_plot,
    # )

    # plot_recurrence_grid_joint_embedding(
    #     args,
    #     model_recurrent,
    #     imgs,
    #     labels,
    #     layer_categories,
    #     steps=model_recurrent.times,
    #     max_steps=max_steps_to_plot,
    # )

    plot_cache_models_joint_embedding(
        args,
        imgs,
        labels,
        layer_categories,
        cache_dir="blt_local_cache",
        max_steps=max_steps_to_plot,
    )

    # visualization of model
    # viz = visualize_blt(model_recurrent)
    # from IPython.display import display

    # viz = TikzComputationGraphVisualizer(model_recurrent)
    # display(viz)


if __name__ == "__main__":

    main()