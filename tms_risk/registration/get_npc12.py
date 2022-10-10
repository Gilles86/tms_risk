import os.path as op
import argparse
from get_npc_mask import main as get_npc
from nilearn import image

def main(subject, bids_folder):

    get_npc(subject, bids_folder=bids_folder, roi='NPC1')
    get_npc(subject, bids_folder=bids_folder, roi='NPC2')

    npc1 = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-T1w_desc-NPC1r_mask.nii.gz')
    npc2 = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-T1w_desc-NPC2r_mask.nii.gz')

    npc12 = image.math_img('npc1+npc2', npc1=npc1, npc2=npc2)
    npc12.to_filename(op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-T1w_desc-NPC12r_mask.nii.gz'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.subject, args.bids_folder)
