#import kagglehub
import kaggle

# Download latest version
# path = kagglehub.competition_download('carvana-image-masking-challenge')
data_path = "/home/ptf/dl-pet-data/carvana/"

kaggle.api.authenticate()
for f in ["train_masks.csv.zip"]:
    kaggle.api.competition_download_file("carvana-image-masking-challenge", f, path=data_path)
# # только train_hq subset + маски (это ~5 ГБ вместо 24)
# kaggle competitions download -c carvana-image-masking-challenge -f train_masks.zip
# kaggle competitions download -c carvana-image-masking-challenge -f train_masks.csv.zip
# kaggle competitions download -c carvana-image-masking-challenge -f metadata.csv.zip

