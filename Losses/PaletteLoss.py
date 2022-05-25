import torch
from torch import nn, optim

from util import palette_from_string
from Losses.LossInterface import LossInterface


class PaletteLoss(LossInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def add_settings(parser):
        parser.add_argument(
            "--palette_weight",
            type=float,
            help="strength of pallete loss effect",
            default=1,
            dest="palette_weight",
        )
        return parser

    def parse_settings(self, args):
        return args

    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        target_palette = (
            torch.FloatTensor(args.palette).requires_grad_(False).to(self.device)
        )
        all_loss = []
        for _, cutouts in cur_cutouts.items():
            _pixels = cutouts.permute(0, 2, 3, 1).reshape(-1, 3)
            palette_dists = torch.cdist(target_palette, _pixels, p=2)
            best_guesses = palette_dists.argmin(axis=0)
            diffs = _pixels - target_palette[best_guesses]
            palette_loss = torch.mean(torch.norm(diffs, 2, dim=1)) * cutouts.shape[0]
            all_loss.append(palette_loss * args.palette_weight / 10.0)
        return all_loss


class ResmemLoss(LossInterface):
    def __init__(self, **kwargs):
        if not os.path.exists(resmem.path):
            wget_file(resmem_url, resmem.path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(**kwargs)
        self.model = ResMem(pretrained=True).to(self.device)
        self.model.eval()

    @staticmethod
    def add_settings(parser):
        parser.add_argument(
            "--symmetry_weight",
            type=float,
            help="how much symmetry is weighted in loss",
            default=1,
            dest="symmetry_weight",
        )
        return parser

    def get_loss1(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        device = self.device
        image_x = rcenter(out)
        prediction = self.model(image_x.view(-1, 2, 227, 227))
        the_loss = 1.0 - prediction[0][0]

        return the_loss

    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        device = self.device
        images = cur_cutouts[224]
        image_x = recenter(images)
        prediction = self.model(image_x)
        mean = torch.mean(prediction)
        mapped_mean = map_number(mean, 0.4, 1.0, 0, 1)

        the_loss = 0.05 * mapped_mean

        return the_loss
