import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_images(log_dir, output_dir=None):
    """
    Extracts all images from a TensorBoard log directory and saves them as PNGs.
    """
    if output_dir is None:
        output_dir = os.path.join(log_dir, "extracted_images")
    os.makedirs(output_dir, exist_ok=True)

    event_acc = EventAccumulator(log_dir, size_guidance={'images': 0})
    event_acc.Reload()

    image_tags = event_acc.Tags()['images']
    print("Available image tags:", image_tags)

    for tag in image_tags:
        events = event_acc.Images(tag)
        for i, event in enumerate(events):
            img_data = event.encoded_image_string
            fname = f"{tag.replace('/', '_')}_step{event.step}.png"
            with open(os.path.join(output_dir, fname), "wb") as f:
                f.write(img_data)
        print(f"Saved {len(events)} images for tag '{tag}' to {output_dir}")

if __name__ == "__main__":
    # Example usage: update the log_dir as needed
    # runs/KLAxb_reconstruction_ML/YYYYMMDD-HHMMSS
    log_dir = "../runs/KLAxb_reconstruction_SL/20250613-164225"
    extract_images(log_dir)