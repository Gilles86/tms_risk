import os.path as op
import argparse
import glob
import shutil


def main(subject, bids_folder='/data'):
    sourcedata_root = op.join(bids_folder, 'sourcedata', 'behavior', f'sub-{subject}', 
    f'ses-1')

    old_subject = int(subject) - 100

    fns = glob.glob(op.join(sourcedata_root, f'sub-{old_subject}*'))

    for fn in fns:
        shutil.move(fn, fn.replace(f'sub-{old_subject}', f'sub-{subject}'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)
