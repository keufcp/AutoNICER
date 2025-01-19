import queue
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import os
import sys
import glob
from PIL import Image
import threading

# from nicer import NICER
from neural_models import error_callback, CAN
from ReLIC.models.relic2_model import NIMA
import config
from autobright import normalize_brightness
from neural_models import error_callback, CAN, NIMA_VGG
from utils import nima_transform, print_msg, loss_with_l2_regularization, monitor_enhance_progress

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Using", device)

class NICER(nn.Module):

    def __init__(self, checkpoint_can, checkpoint_nima, device='cpu', can_arch=13, aesthetic_model="NIMA"):
        super(NICER, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using", self.device)

        # if can_arch != 8 and can_arch != 7:
        #     error_callback('can_arch')

        can = CAN(can_arch)
        can.load_state_dict(torch.load(checkpoint_can, map_location=self.device))
        can.eval()
        can.to(self.device)

        # aesthetic_model = "ReLIC"
        self.aesthetic_model = aesthetic_model

        if aesthetic_model == "ReLIC":
            #! 変更点
            relic = NIMA()
            relic.load_state_dict(torch.load("ReLIC/pretrain_model/relic2_model.pth", map_location=self.device))
            relic.eval()
            relic.to(self.device)
        elif aesthetic_model == "NIMA":
            relic = NIMA_VGG(models.vgg16(pretrained=True)) #! 注意！！NIMA読み込み
            relic.load_state_dict(torch.load(checkpoint_nima, map_location=self.device))
            relic.eval()
            relic.to(self.device)
        else:
            print("Please check the aesthetic_model.")
            exit(0)
            
        torch.autograd.set_detect_anomaly(True)
        self.queue = queue.Queue()

        # self.filters is a leaf-variable, bc it's created directly and not as part of an operation
        self.filters = torch.tensor([0.0 for _ in range(can_arch)], dtype=torch.float32, requires_grad=True,
                                    device=self.device) #! フィルタ数を可変に変更 
        self.can = can
        self.relic = relic

        self.gamma = config.gamma

        self.can_arch = can_arch

        if config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)
        elif config.optim == 'adam':
            self.optimizer = torch.optim.Adam(params=[self.filters], lr=config.optim_lr)
        else:
            error_callback('optimizer')

    def forward(self, image, fixedFilters=None):
        filter_tensor = torch.zeros((self.can_arch, 224, 224), dtype=torch.float32).to(self.device)

        # construct filtermap uniformly from given filters
        for l in range(self.can_arch):
            if fixedFilters:  # filters were fixed in GUI, always use their passed values
                if fixedFilters[l][0] == 1:
                    filter_tensor[l, :, :] = fixedFilters[l][1]
                else:
                    filter_tensor[l, :, :] = self.filters.view(-1)[l]
            else:
                filter_tensor[l, :, :] = self.filters.view(-1)[l]

        mapped_img = torch.cat((image, filter_tensor.cpu()), dim=0).unsqueeze(dim=0).to(
            self.device)  # concat filters and img
        enhanced_img = self.can(mapped_img)  # enhance img with CAN
        distr_of_ratings = self.relic(enhanced_img)  # get relic score distribution -> tensor

        self.queue.put('dummy')  # dummy

        return distr_of_ratings, enhanced_img

    def set_filters(self, filter_list):
        # usually called from GUI
        if max(filter_list) > 1:
            filter_list = [x / 100.0 for x in filter_list]

        with torch.no_grad():
            for i in range(5):
                self.filters[i] = filter_list[i]
            self.filters[5] = filter_list[6]  # llf is 5 in can but 6 in gui (bc exp is inserted)
            self.filters[6] = filter_list[7]  # nld is 6 in can but 7 in gui
            self.filters[7] = filter_list[5]  # exp is 7 in can but 5 in gui

    def set_gamma(self, gamma):
        self.gamma = gamma

    def single_image_pass_can(self, image, abn=False, filterList=None, mapToCpu=False):
        """
            pass an image through the CAN architecture 1 time. This is usually called from the GUI, to preview the images.
            It is also called when the image is to be saved, since we then need to apply the final filter intensities onto the image.

            if called_to_save_raw is False, this method will return an 8bit image to show what the current filter combination looks
            like (because PIL cannot process 16bit). If called_to_save_raw is true, it will return the enhanced 16bit image as
            np.uint16 array, to be saved with opencv.imwrite() as 16bit png.
        """

        # filterList is passable as argument because for previwing the imgs in the GUI while optimizing,
        # we cannot use self.filters, as this is used for gradient computatation

        device = self.device if mapToCpu is False else 'cpu'

        if abn:
            bright_norm_img = normalize_brightness(image, input_is_PIL=True)
            image = Image.fromarray(bright_norm_img)

        if image.size[1] > config.final_size or image.size[0] > config.final_size:
            image_tensor = transforms.Compose([
                transforms.Resize(config.final_size),
                transforms.ToTensor()])(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        filter_tensor = torch.zeros((self.can_arch, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32).to(
            device)  # tensorshape [c,w,h]
        for l in range(self.can_arch):
            filter_tensor[l, :, :] = filterList[l] if filterList else self.filters.view(-1)[l]

        mapped_img = torch.cat((image_tensor.cpu(), filter_tensor.cpu()), dim=0).unsqueeze(dim=0).to(device)

        try:
            enhanced_img = self.can(mapped_img)  # enhance img with CAN
        except RuntimeError:
            self.can.to('cpu')
            try:
                enhanced_img = self.can(mapped_img)  # enhance img with CAN
            except RuntimeError:
                print("DefaultCPUAllocator - not enough memory to perform this operation")
                return None
            self.can.to('cuda')

        enhanced_img = enhanced_img.cpu()
        enhanced_img = enhanced_img.detach().permute(2, 3, 1, 0).squeeze().numpy()

        enhanced_clipped = np.clip(enhanced_img, 0.0, 1.0) * 255.0
        enhanced_clipped = enhanced_clipped.astype('uint8')

        # returns a np.array of type np.uint8

        return enhanced_clipped

    def re_init(self):
        # deprecated, formerly used for batch mode
        self.filters = torch.tensor([0.0 for _ in range(self.can_arch)], dtype=torch.float32, requires_grad=True,
                                    device=self.device)
        if config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)
        elif config.optim == 'adam':
            self.optimizer = torch.optim.Adam(params=[self.filters], lr=config.optim_lr)
        else:
            error_callback('optimizer')

    def enhance_image(self, image_path, re_init=True, fixFilters=None, epochs=config.epochs, thread_stopEvent=None):
        """
            optimization routine that is called to enhance an image.
            Usually this is called from the NICER button in the GUI.
            Accepts image path as a string, but also as PIL image.

            Returns a re-sized 8bit image as np.array
        """

        if re_init:
            self.re_init()
        else:
            # re-init is false, i.e. use user_preset filters that are selected in the GUI
            # re-init can be seen as test whether initial filter values (!= 0) should be used or not during optimization
            user_preset_filters = [self.filters[x].item() for x in range(self.can_arch)]

        if isinstance(image_path, str):
            bright_normalized_img = normalize_brightness(image_path)
            pil_image = Image.fromarray(bright_normalized_img)
        else:
            pil_image = image_path
            bright_normalized_img = normalize_brightness(pil_image, input_is_PIL=True)
            pil_image = Image.fromarray(bright_normalized_img)

        image_tensor_transformed = nima_transform(pil_image)
        image_tensor_for_distri = image_tensor_transformed.unsqueeze(0).to(self.device)

        distribution = self.relic(transforms.Resize((224, 224), antialias=False)(image_tensor_for_distri)) #! 追加
        #! スコア表示追加
        list_distri = distribution.tolist()[0] # [0]で外側の[]を取り除く
        first_approx_score = sum((i+1) * prob for i, prob in enumerate(list_distri))
        print_msg(f"First approx score: {first_approx_score} ({self.aesthetic_model})", 3)
        # print_msg(f"Score distribution: {list_distri}", 3) #? 分布を表示

        if fixFilters:  # fixFilters is bool list of filters to be fixed
            initial_filter_values = []
            for k in range(self.can_arch):
                if fixFilters[k] == 1:
                    initial_filter_values.append([1, self.filters[k].item()])
                else:
                    initial_filter_values.append([0, self.filters[k].item()])

        # optimize image:
        print_msg("Starting optimization", 2)
        start_time = time.time()
        score_list = []
        for i in range(epochs):
            # if thread_stopEvent.is_set(): break

            print_msg("Iteration {} of {}".format(i, epochs), 2)

            if fixFilters:
                distribution, enhanced_img = self.forward(image_tensor_transformed, fixedFilters=initial_filter_values)
            else:
                distribution, enhanced_img = self.forward(image_tensor_transformed)

            self.optimizer.zero_grad()

            if re_init:
                # new for each image
                loss = loss_with_l2_regularization(distribution.cpu(), self.filters.cpu(), gamma=self.gamma)
            else:
                loss = loss_with_l2_regularization(distribution.cpu(), self.filters.cpu(),
                                                   initial_filters=user_preset_filters, gamma=self.gamma)
            
            #! スコア表示追加
            list_distri = distribution.tolist()[0] # [0]で外側の[]を取り除く
            approx_score = sum((i+1) * prob for i, prob in enumerate(list_distri))
            print_msg(f"Approx score: {approx_score} ({self.aesthetic_model})", 3)
            score_list.append(approx_score)
            # print_msg(f"Score distribution: {list_distri}", 3) #? 分布を表示

            loss.backward()
            self.optimizer.step()

            filters_for_queue = [self.filters[x].item() for x in range(self.can_arch)]
            self.queue.put(i + 1)
            self.queue.put(filters_for_queue)

            epoch_count = i
            # if monitor_enhance_progress(score_list, 10) == False: break
            
            # ファイルパスからディレクトリ/ファイル名を取得
            # directory = os.path.dirname(image_path)
            # filename, _ = os.path.splitext(os.path.basename(image_path))

            # if not os.path.isdir(f"{directory}\{filename}"):
            #     os.makedirs(f"{directory}\{filename}")

            # enhanced_clipped_np = self.single_image_pass_can(pil_image, abn=False, filterList=filters_for_queue)
            # hd_img_pil = Image.fromarray(enhanced_clipped_np)

            # hd_img_pil.save(f"{directory}\{filename}\epoch_{epoch_count}.png")
            
            # f = open(f"{directory}\{filename}\scores_{filename}_{self.aesthetic_model}.csv", 'a')
            # print(f"{epoch_count}, {approx_score}", file=f)
            # g = open(f"{directory}\{filename}\prams_{self.can_arch}_{filename}.csv", 'a')
            # param_list = self.filters.tolist()
            # print(epoch_count, *param_list, sep=', ', file=g)

        if not thread_stopEvent.is_set():
            print_msg("Optimization for %d epochs took %.3fs" % (epoch_count, time.time() - start_time), 2)

            # the entire rescale thing is not needed, bc optimization happens on a smaller image (for speed improvement)
            # real rescale is done during saving.
            original_tensor_transformed = transforms.ToTensor()(pil_image)

            final_filters = torch.zeros((self.can_arch, original_tensor_transformed.shape[1], original_tensor_transformed.shape[2]),
                                        dtype=torch.float32).to(self.device)
            for k in range(self.can_arch):
                if fixFilters:
                    if fixFilters[k] == 1:
                        final_filters[k, :, :] = initial_filter_values[k][1]
                    else:
                        final_filters[k, :, :] = self.filters.view(-1)[k]
                else:
                    final_filters[k, :, :] = self.filters.view(-1)[k]

            strings = ['Sat', 'Con', 'Bri', 'Sha', 'Hig', 'LLF', 'NLD', 'EXP', 'Mid', 'temp', 'chmxred', 'chmxgreen', 'chmxblue']
            print_msg("Final Filter Intensities: {}".format(
                [strings[k] + ": " + str(final_filters[k, 0, 0].item() * 100) for k in range(self.can_arch)]), 3)
            self.queue.put([final_filters[k, 0, 0].item() for k in range(self.can_arch)])

            mapped_img = torch.cat((original_tensor_transformed, final_filters.cpu()), dim=0).unsqueeze(dim=0).to(
                self.device)
            enhanced_img = self.can(mapped_img)
            distribution = self.relic(transforms.Resize((224, 224), antialias=False)(enhanced_img)) #! 追加
            enhanced_img = enhanced_img.cpu()
            enhanced_img = enhanced_img.detach().permute(2, 3, 1, 0).squeeze().numpy()
            enhanced_clipped = np.clip(enhanced_img, 0.0, 1.0) * 255.0
            enhanced_clipped = enhanced_clipped.astype('uint8')

            #! スコア表示追加
            list_distri = distribution.tolist()[0] # [0]で外側の[]を取り除く
            approx_score = sum((i+1) * prob for i, prob in enumerate(list_distri))
            print_msg(f"Final approx score: {approx_score} ({self.aesthetic_model}) (First approx score: {first_approx_score})", 3)
            # print_msg(f"Score distribution: {list_distri}", 3) #? 分布を表示

            self.queue.put(enhanced_clipped)

            # returns an 8bit image in any case ---
            return enhanced_clipped, [final_filters[k, 0, 0].item() for k in range(self.can_arch)], None

def batch_imag_enhancement(imgs_path: str, model="NIMA"):

    can_path = r"user_models\can_model_2023-12-12_07-53-41.pt" #! 適宜変更!!
    src_img_path = f"{imgs_path}"

    nicer = NICER(checkpoint_can=can_path, checkpoint_nima=config.nima_checkpoint_path, can_arch=13, aesthetic_model=model)

    if not imgs_path:
        print("Specify the directory for the images.")
        exit(0)
    
    # if not os.path.isdir(f"{imgs_path}\dst"):
    #     os.makedirs(f"{src_img_path}\dst")
    
    img_names_jpg = glob.glob(src_img_path + "\*.jpg")
    img_names_png = glob.glob(src_img_path + "\*.png")
    # img_ids = [os.path.basename(file_path) for file_path in img_names]
    img_ids = img_names_jpg + img_names_png
    print(img_ids)
    
    for i in range(len(img_ids)):
        print(f"Process {img_ids[i]}")
        # input_img = Image.open(f"{img_ids[i]}")
        # img_tensor = transforms.ToTensor()(input_img)
        # img_tensor = transforms.Resize(size=1080, antialias=False)(img_tensor).to(device) #? antialias=False ← 実行時警告対策

        enhanced_clipped, _, _ = nicer.enhance_image(image_path=img_ids[i], re_init=True, thread_stopEvent=threading.Event())

if __name__ == '__main__':
    batch_imag_enhancement(sys.argv[1], sys.argv[2])