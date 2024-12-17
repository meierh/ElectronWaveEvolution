# generated by ChatGPT
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(filename, scale, dpi=300, output_format='pdf', simple=False, show_title=True):
    # Read the CSV file
    data = pd.read_csv(filename)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Set axis scales based on the scale argument
    if scale == 'log':
        plt.xscale('log')
        plt.yscale('log')
        plot_title = f'Log-Scale Plot of {filename}'
    elif scale == 'lin':
        plot_title = f'Linear-Scale Plot of {filename}'
    else:
        raise ValueError("Invalid scale value. Use 'log' or 'lin'.")

    # Set labels
    plt.xlabel(data.columns[0])  # Set the x-axis label from the CSV column name
    plt.ylabel(data.columns[1])  # Set the y-axis label from the CSV column name

    # Set title only if show_title is True
    if show_title:
        plt.title(plot_title)  # Set the title using the filename and scale

    plt.grid(True)

    if simple:
        # Plot data as a connected line without legend or numbering
        plt.plot(data.iloc[:, 0], data.iloc[:, 1], 'b-o')  # Blue line with circle markers
    else:
        # Plot each data point individually with its number in the legend
        for i in range(len(data)):
            x = data.iloc[i, 0]
            y = data.iloc[i, 1]
            
            # Plot each point and assign a unique label for the legend
            label = f'{i+1}: ({x}, {y})'  # Format the label for the legend
            plt.plot(x, y, 'bo', label=label)  # Plot with blue dots

            # Annotate the points with numbers
            plt.text(x * 1.1, y * 1.1, f'{i+1}', fontsize=8, ha='left', va='bottom')  # Annotate with index number

        # Remove duplicate labels in the legend (so each point has a unique label in the legend)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        seen_labels = set()
        for handle, label in zip(handles, labels):
            if label not in seen_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
                seen_labels.add(label)

        # Create the legend with unique labels
        plt.legend(unique_handles, unique_labels, title="Data Points", loc='upper left', fontsize=4)

    # Save the plot to a file in the desired format
    output_filename = filename.split('.')[0] + f'_{scale}_plot' + ('_simple' if simple else '_detailed') + f'.{output_format}'
    if output_format == 'png':
        plt.savefig(output_filename, dpi=dpi)  # Use DPI for raster formats
    else:
        plt.savefig(output_filename)  # Vectors don't need DPI
    print(f'{scale.capitalize()}-Scale Plot saved as {output_filename}')
    plt.close()

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Plot and save data from a CSV file with configurable scales (log or lin) and optional simplification.')
    parser.add_argument('filename', type=str, help='Name of the CSV file to plot')
    parser.add_argument('--scale', type=str, choices=['log', 'lin'], default='log',
                        help="Scale for the plot. Use 'log' for logarithmic (default) or 'lin' for linear.")
    parser.add_argument('--dpi', type=int, default=300, help='DPI for the output image (default: 300 for PNG).')
    parser.add_argument('--format', type=str, choices=['pdf', 'svg', 'png'], default='pdf',
                        help="Output format for the plot. Default: 'pdf'. Options: 'pdf', 'svg', 'png'.")
    parser.add_argument('--simple', action='store_true', help='Produce a simple plot without numbering and legend, connecting data points.')
    parser.add_argument('--notitle', action='store_true', help='Disable the title of the plot.')
    args = parser.parse_args()

    # Call the function to plot the data from the provided CSV file
    plot_csv(args.filename, args.scale, dpi=args.dpi, output_format=args.format, simple=args.simple, show_title=not args.notitle)
