import os
import sys


def extract_name(files):
    ## extract frame number from folders
    frame_list = []
    for f in files:
        if f.endswith('png'):
            frame_list.append(int(f.split('.')[0].split('frame_')[1]))
    return frame_list


def score(ans_list, result_list):
    ## initiate score values
    hit = 1
    miss = len(ans_list)

    for x in result_list:
        inter_x = range(x-10, x+11)
        if any(x in ans_list for x in inter_x):
            hit = hit + 1

    for y in ans_list:
        inter_y = range(y-10, y+11)
        if any(y in result_list for y in inter_y):
            miss = miss - 1

    print 'hit %s out of %s' % (hit, len(result_list)+1)
    print 'miss %s out of %s' % (miss, len(ans_list)+1)


def main():

    if len(sys.argv) < 2:
        print "Error - file name must be specified as first argument."
        return

    file_id = sys.argv[1]
    files = os.listdir('ans/%s/' % (file_id))
    ans_list = extract_name(files)
    files = os.listdir('naiive/%s/' % (file_id))
    result_list = extract_name(files)

    score(ans_list, result_list)


if __name__ == "__main__":
    main()
