import os
import numpy as np
import torchvision.transforms as transforms
from preprocessing.randaugment import randaugmentation
from preprocessing.randaugment.policy import randaug_policies
from tqdm import tqdm
from PIL import Image

num_copy = 4
run_on = "CRC_DX"

def read_and_split(data_dir, destinate_dir, dataset):
  print("\n>> Processing %s" % data_dir)
  if os.path.isdir(data_dir):
    # Read from the file
    all_files = os.listdir(data_dir)
    np.random.shuffle(all_files)
    mean, std = randaugmentation.get_mean_and_std()

    for file_name in tqdm(all_files):
      # Only accept the jpg file
      if ".png" in file_name:
        image = Image.open(os.path.join(data_dir, file_name))
        image = np.array(image.convert('RGB')) / 255.0
        image = (image - mean) / std

        for index in range(num_copy):
          aug_choices = np.random.choice(len(aug_policies))
          chosen_policy = aug_policies[aug_choices]
          aug_image = randaugmentation.apply_policy(chosen_policy, image, dataset)
          aug_image = randaugmentation.cutout_numpy(aug_image)
          aug_image = randaugmentation.pil_wrap(aug_image, dataset, use_mean_std = True).convert("RGB")
          aug_image.save(os.path.join(destinate_dir, \
            file_name.split(".png")[0] + "-" + str(index) + ".png"))

    print(">> Total running {}".format(len(all_files)))


# def fix_file():
#   data_dir = os.path.join(os.getcwd(), "data", "augmentation")
#   all_files = os.listdir(data_dir)

#   for file_name in tqdm(all_files, total = len(all_files)):
#     temp = file_name.split(".png")
#     new_file_name = "".join(temp + [".png"])

#     os.rename(os.path.join(data_dir, file_name), os.path.join(data_dir, new_file_name))

if __name__ == "__main__":
  aug_policies = randaug_policies()
  read_and_split("data/{}/train/msimut".format(run_on), "data/{}/augmentation".format(run_on))
  read_and_split("data/{}/train/mss".format(run_on), "data/{}/augmentation".format(run_on))