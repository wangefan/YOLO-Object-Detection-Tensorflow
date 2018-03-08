classes_name =  ["license"]

# 根據classes_name建立index classes_no = [0, 1,....]
classes_no = [i for i in range(len(classes_name))]

# classes_dict = [{"aeroplane", 0}, {"bicycle", 1},...]
classes_dict = dict(zip(classes_name, classes_no))
num_class = len(classes_name)

# got {normal, fast}
# but if you change the type, u need to train first for atleast for few hours
model_type = 'fast'

image_size = 448
cell_size = 7
box_per_cell = 2
alpha_relu = 0.1
object_scale = 2.0
no_object_scale = 1.0
class_scale = 2.0
coordinate_scale = 5.0
flipped = False
data_set_path = '.\data\lpr_sq'
cache_file_path = '.\data\cache'
decay_step = 30000
decay_rate = 0.1
learning_rate = 0.0001
dropout = 0.5
batch_size = 3
epoch = 1000
checkpoint = 1000

# For main
threshold = 0.2
IOU_threshold = 0.5
test_percentage = 0.1

# 1 for read a picture
# 2 to read from testing dataset
# 3 to read from webcam / video
output = 2
# let empty if want to capture from webcam
picture_name = ''
video_name = ''


