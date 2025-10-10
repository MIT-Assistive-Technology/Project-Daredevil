from ultralytics import YOLO
from PIL import Image


# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run inference on an image
results = model("https://ultralytics.com/images/bus.jpg")  # results list

# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")