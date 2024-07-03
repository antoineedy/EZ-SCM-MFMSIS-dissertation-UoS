from models.segmentor.zegclip import ZegCLIP

from models.segmentor.xx_scales_input_zegclip import MultiScales
from models.segmentor.xx_scales_output_zegclip import MultiScalesOutputZegCLIP
from models.segmentor.inner_zegclip import InnerZegCLIP

from models.other_modules.multi_scale import MultiScales

from models.backbone.text_encoder import CLIPTextEncoder
from models.backbone.img_encoder import CLIPVisionTransformer, VPTCLIPVisionTransformer
from models.decode_heads.decode_seg import ATMSingleHeadSeg

from models.losses.atm_loss import SegLossPlus

from configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
from configs._base_.datasets.dataloader.coco_stuff import ZeroCOCOStuffDataset
