import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 100000000

class IOULoss(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, ):
        return 

def clip_sigmoid(input):
    out = torch.clamp(torch.sigmoid(input), min=1e-4, max=1 - 1e-4)

    return out


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        # print(term1.sum(), term2.sum())

        loss = (
            -(t == class_ids).float() * alpha * term1
            - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        return loss.sum()


class FCOSLoss(nn.Module):
    def __init__(self, sizes, gamma, alpha, center_sampling, fpn_strides, pos_radius):
        super().__init__()
        self.sizes = sizes
        self.gamma = gamma
        self.alpha = alpha
        self.center_sampling = center_sampling
        self.fpn_strides = fpn_strides
        self.radius = pos_radius

    def forward(self, locations, preds, targets):
        pred_logits, pred_bboxes, pred_centers = preds
        labels, box_targets = self.prepare_targets(locations, targets)

        cls_flat = []
        box_flat = []
        center_flat = []

        labels_flat = []
        box_targets_flat = []

        batch_size = len(pred_logits)
        num_class = pred_logits[0].shape[1]

        for i in range(len(labels)):
            cls_flat.append(pred_logits[i].permute(0, 2, 3, 1).reshape(-1, num_class))
            box_flat.append(pred_bboxes[i].permute(0, 2, 3, 1).reshape(-1, 4))
            center_flat.append(pred_centers[i].permute(0, 2, 3, 1).reshape(-1))

            labels_flat.append(labels[i].reshape(-1))
            box_targets_flat.append(box_targets[i].reshape(-1, 4))

        cls_flat = torch.cat(cls_flat, 0)
        box_flat = torch.cat(box_flat, 0)
        center_flat = torch.cat(center_flat, 0)

        labels_flat = torch.cat(labels_flat, 0)
        box_targets_flat = torch.cat(box_targets_flat, 0)

        import pdb; pdb.set_trace()
        pos_id = torch.nonzero(labels_flat > 0).squeeze(1)

        cls_loss = self.cls_loss(cls_flat, labels_flat.int()) / (pos_id.numel() + batch)

        return 

    def prepare_targets(self, locations, targets):
        ex_size_of_interest = []
        for i, location_per_level in enumerate(locations):
            size_of_interest_per_level = location_per_level.new_tensor(self.sizes[i], )
            ex_size_of_interest.append(
                size_of_interest_per_level[None].expand(len(location_per_level), -1)
            )
        
        ex_size_of_interest = torch.cat(ex_size_of_interest, 0)
        n_location_per_level = [len(location_per_level) for location_per_level in locations]
        all_locations = torch.cat(locations, dim=0)
        all_locations = all_locations.to(targets.device)

        label, box_target = self.computer_target_for_location(
            all_locations, targets, ex_size_of_interest, n_location_per_level
        )

        for i in range(len(label)):
            label[i] = torch.split(label[i], n_location_per_level, 0)
            box_target[i] = torch.split(box_target[i], n_location_per_level, 0)

        label_level_first = []
        box_target_level_first = []

        for level in range(len(locations)):
            label_level_first.append(
                torch.cat([label_per_img[level] for label_per_img in label], 0)
            )
            box_target_level_first.append(
                torch.cat(
                    [box_target_per_img[level] for box_target_per_img in box_target], 0
                )
            )

        return label_level_first, box_target_level_first

    def get_sample_region(self, boxes, strides, n_location_per_level, xs, ys, radius=1):
        n_gt = boxes.shape[0]
        n_loc = len(xs)
        box_xyxy = torch.zeros_like(boxes)
        box_xyxy[:, :2] = boxes[:, :2]
        box_xyxy[:, 2:4] = boxes[:, :2] + boxes[:, 2:]

        box_xyxy = box_xyxy[None].expand(n_loc, n_gt, 4)
        center_x = (box_xyxy[..., 0] + box_xyxy[..., 2]) / 2
        center_y = (box_xyxy[..., 1] + box_xyxy[..., 3]) / 2

        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        begin = 0
        
        center_gt = box_xyxy.new_zeros(box_xyxy.shape)
        for level, num_loc in enumerate(n_location_per_level):
            end = begin + num_loc

            stride = strides[level] * radius

            x_min = center_x[begin:end] - stride
            y_min = center_y[begin:end] - stride
            x_max = center_x[begin:end] + stride
            y_max = center_y[begin:end] + stride

            center_gt[begin:end, :, 0] = torch.where(
                x_min > box_xyxy[begin:end, :, 0], x_min, box_xyxy[begin:end, :, 0]
            )
            center_gt[begin:end, :, 1] = torch.where(
                y_min > box_xyxy[begin:end, :, 1], y_min, box_xyxy[begin:end, :, 1]
            )
            center_gt[begin:end, :, 2] = torch.where(
                x_max > box_xyxy[begin:end, :, 2], box_xyxy[begin:end, :, 2], x_max
            )
            center_gt[begin:end, :, 3] = torch.where(
                y_max > box_xyxy[begin:end, :, 3], box_xyxy[begin:end, :, 3], y_max
            )

            begin = end
        
        left = xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - xs[:, None]
        top = ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - ys[:, None]

        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_boxes = center_bbox.min(-1)[0] > 0

        return is_in_boxes

    def computer_target_for_location(self, locations, targets, sizes_of_interest, n_location_per_level):
        labels = []
        box_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for i in range(len(targets)):
            target_per_img = targets[i]
            # Box shape is x,y, w, h
            boxes = target_per_img[:, :4]
            labels_per_img = target_per_img[:, 4]
            area = torch.mul(boxes[:, 2], boxes[:, 3])

            left = xs[:, None] - boxes[:, 0][None]
            top = ys[:, None] - boxes[:, 1][None]
            right = xs[:, None] + boxes[:, 2][None] / 2
            bottom = ys[:, None] + boxes[:, 3][None] / 2

            box_targets_per_image = torch.stack([left, top, right, bottom], 2)
            if self.center_sampling:
                is_in_boxes = self.get_sample_region(
                    boxes, self.fpn_strides, n_location_per_level, xs, ys, radius=self.radius
                )
            else:
                is_in_boxes = box_targets_per_image.min(2)[0] > 0

            max_box_targets_per_img = box_targets_per_image.max(2)[0]

            is_cared_in_level = (
                max_box_targets_per_img >= sizes_of_interest[:, [0]]
            ) & (max_box_targets_per_img <= sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_level == 0] = INF

            locations_to_min_area, locations_to_gt_id = locations_to_gt_area.min(1)

            box_targets_per_image = box_targets_per_image[
                range(len(locations)), locations_to_gt_id
            ]
            labels_per_img = labels_per_img[locations_to_gt_id]
            labels_per_img[locations_to_min_area == INF] = 0

            labels.append(labels_per_img)
            box_targets.append(box_targets_per_image)
        return labels, box_targets
