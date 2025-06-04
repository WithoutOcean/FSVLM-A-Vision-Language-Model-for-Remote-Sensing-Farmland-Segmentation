import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.confusion_matrix_giou = np.zeros((self.num_class,) * 2)
        self.confusion_matrix_ciou = np.zeros((self.num_class,) * 2)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def calculate_iou(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / union
        dice=2*intersection/(np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0))
        recall=intersection/np.sum(self.confusion_matrix, axis=0)
        return iou.mean(), dice.mean(),recall.mean() # 取平均值以确保返回标量
    
    def calculate_giou(self):
        iou = self.calculate_iou()
        giou = 1 - iou
        self.confusion_matrix_giou = np.zeros((self.num_class,) * 2)
        return giou

    def calculate_ciou(self):
        giou = self.calculate_giou()
        ciou = giou / (1 - giou)
        self.confusion_matrix_ciou = np.zeros((self.num_class,) * 2)
        return ciou
    

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # def _generate_matrix(self, gt_image, pre_image):
    #     mask = (gt_image >= 0) & (gt_image < self.num_class)
    #     label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
    #     count = np.bincount(label, minlength=self.num_class**2)
    #     confusion_matrix = count.reshape(self.num_class, self.num_class)
    #     return confusion_matrix
    #
    # def calculate_iou(self, confusion_matrix):
    #     intersection = np.diag(confusion_matrix)
    #     union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - intersection
    #     iou = intersection / union
    #     return iou
    #
    # def calculate_giou(self, confusion_matrix):
    #     iou = self.calculate_iou(confusion_matrix)
    #     giou = 1 - iou
    #     self.confusion_matrix_giou = np.zeros((self.num_class,) * 2)
    #     return giou
    #
    # def calculate_ciou(self, confusion_matrix):
    #     giou = self.calculate_giou(confusion_matrix)
    #     ciou = giou / (1 - giou)
    #     self.confusion_matrix_ciou = np.zeros((self.num_class,) * 2)
    #     return ciou

        # intersection = np.logical_and(gt_image, pre_image)
        # union = np.logical_or(gt_image, pre_image)
        # enclosed_area = np.sum(union)
        #
        # iou = np.sum(intersection) / np.sum(union)
        #
        # centroid_gt = np.mean(np.where(gt_image == 1), axis=1)
        # centroid_pre = np.mean(np.where(pre_image == 1), axis=1)
        # center_distance = np.linalg.norm(centroid_gt - centroid_pre)
        #
        # area_gt = np.sum(gt_image)
        # area_pre = np.sum(pre_image)
        # diagonal_length = np.linalg.norm(gt_image.shape)
        #
        # aspect_ratio_term = 4 / (np.pi ** 2) * ((np.arctan(area_gt / area_pre) - np.arctan(area_pre / area_gt)) ** 2)
        #
        # iou_loss = iou - ((center_distance ** 2) / (diagonal_length ** 2) + aspect_ratio_term)
        #
        # giou = iou - iou_loss
        #
        # c_x_gt, c_y_gt = centroid_gt
        # # print("centroid_gt:", centroid_gt)
        # # if len(centroid_gt) >= 2:
        # #     c_x_gt, c_y_gt = centroid_gt[:2]
        # # else:
        # #     # 处理centroid_gt不足两个值的情况
        # #     print("Unexpected centroid_gt:", centroid_gt)
        # #     # 在这里进行适当的操作，例如赋予默认值或引发错误。
        #
        # c_x_pre, c_y_pre = centroid_pre
        # w_gt, h_gt = np.where(gt_image == 1)[0].max() - np.where(gt_image == 1)[0].min(), np.where(gt_image == 1)[
        #     1].max() - np.where(gt_image == 1)[1].min()
        # w_pre, h_pre = np.where(pre_image == 1)[0].max() - np.where(pre_image == 1)[0].min(), np.where(pre_image == 1)[
        #     1].max() - np.where(pre_image == 1)[1].min()
        #
        # v = 4 / (np.pi ** 2) * np.square(np.arccos((w_gt * h_gt) / (w_pre * h_pre)))
        # alpha = v / (1 - iou + v)
        #
        # ciou = iou - ((center_distance ** 2) / (diagonal_length ** 2) + aspect_ratio_term) + alpha * v
        #
        # return giou, ciou


    def add_batch(self, gt_image, pre_image):
        # assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.confusion_matrix_giou = np.zeros((self.num_class,) * 2)
        self.confusion_matrix_ciou = np.zeros((self.num_class,) * 2)



