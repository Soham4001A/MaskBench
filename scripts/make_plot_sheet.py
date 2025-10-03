import argparse
import os
import glob
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs", help="Root directory containing environment plot folders.")
    args = parser.parse_args()

    output_root_dir = args.output_dir
    if not os.path.isdir(output_root_dir):
        print(f"Error: Output root directory not found: {output_root_dir}")
        return

    env_dirs = sorted([d for d in os.listdir(output_root_dir) if os.path.isdir(os.path.join(output_root_dir, d))])

    if not env_dirs:
        print(f"No environment directories found in {output_root_dir}")
        return

    for env_id in env_dirs:
        env_plots_dir = os.path.join(output_root_dir, env_id)
        print(f"Processing environment: {env_id}")

        plot_files = sorted(glob.glob(os.path.join(env_plots_dir, "*.png")))
        if not plot_files:
            print(f"No plots found in {env_plots_dir}. Skipping.")
            continue

        images = []
        for plot_file in plot_files:
            img = cv2.imread(plot_file)
            if img is None:
                print(f"Warning: Could not load image {plot_file}. Skipping.")
                continue
            images.append(img)

        if not images:
            print(f"No valid images to combine for {env_id}. Skipping.")
            continue

        # Assume all images are of similar size, resize to the first image's dimensions
        target_height, target_width, _ = images[0].shape
        resized_images = [cv2.resize(img, (target_width, target_height)) for img in images]

        # Determine grid dimensions
        num_images = len(resized_images)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))

        # Create an empty canvas for the combined image
        combined_width = cols * target_width
        combined_height = rows * target_height
        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        # Place images in the grid
        for i, img in enumerate(resized_images):
            row_idx = i // cols
            col_idx = i % cols
            y_offset = row_idx * target_height
            x_offset = col_idx * target_width
            combined_image[y_offset:y_offset + target_height, x_offset:x_offset + target_width] = img

        # Save the combined image
        output_file = os.path.join(output_root_dir, f"{env_id}_plot_sheet.png")
        cv2.imwrite(output_file, combined_image)
        print(f"Combined plot sheet for {env_id} saved to {output_file}")

if __name__ == "__main__":
    main()
