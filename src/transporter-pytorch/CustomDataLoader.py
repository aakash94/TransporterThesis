import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

transform = transforms.ToTensor()


def get_image(n, t):
    im = Image.open('{}/{}/{}.png'.format("data", n, t))
    # im.show()
    im = np.array(im)
    # im = transform(im)
    return im


def get_index(v, fv, max_v=84):
    i = int(v + fv)
    i = max(0, min(i, max_v - 1))
    return i


def apply_flow(flow: np.ndarray, prev_i: np.ndarray, next_i: np.ndarray):
    # https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
    y_val, x_val = prev_i.shape
    final_i = np.zeros_like(prev_i)
    for y in range(y_val):
        for x in range(x_val):
            flow_val = flow[y][x]
            final_i[y][x] = next_i[get_index(v=y, fv=flow_val[1], max_v=y_val)] \
                [get_index(v=x, fv=flow_val[0], max_v=x_val)]
    return final_i

def apply_flow2(flow: np.ndarray, prev_i: np.ndarray):
    # https://pyimagesearch.com/2021/02/03/opencv-image-translation/
    avg_motion = np.mean(flow, axis=(0, 1))
    affine_mat = np.float32([
        [1, 0, avg_motion[0]],
        [0, 1, avg_motion[1]]
    ])
    final_i = cv2.warpAffine(prev_i, affine_mat, (prev_i.shape[1], prev_i.shape[0]))
    # print("motion avg", avg_motion)
    # for y in range(y_val):
    #     for x in range(x_val):
    #         flow_val = flow[y][x]
    #         final_i[y][x] = next_i[get_index(v=y, fv=flow_val[1], max_v=y_val)] \
    #             [get_index(v=x, fv=flow_val[0], max_v=x_val)]
    return final_i



def trnsformed_images(i1, i2):
    if not isinstance(i1, np.ndarray):
        i1 = np.array(i1)

    if not isinstance(i2, np.ndarray):
        i2 = np.array(i2)

    i1 = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_RGB2GRAY)
    motion_i = get_motion(img_new=i2, img_old=i1)
    reconstruct = apply_flow(flow=motion_i, prev_i=i1, next_i=i2)
    # print("i1", i1[0])
    # print("i2", i2[0])
    # v0 = motion_i[:, :, 0]
    # v1 = motion_i[:, :, 1]
    # print("m", motion_i)
    # apply flow to i1
    r = Image.fromarray(reconstruct)
    print("reconstructed")
    r.show(title="r")

    return reconstruct, i2


def get_motion(img_new, img_old):
    flow = cv2.calcOpticalFlowFarneback(prev=img_old,
                                        next=img_new,
                                        flow=None,
                                        pyr_scale=0.5,
                                        levels=3,
                                        winsize=15,
                                        iterations=3,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        flags=0)
    '''
    The below code is used to show the flow
    # https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
    mask = np.zeros_like(img_old)
    mask = np.dstack((mask, mask, mask))
    # Sets image saturation to maximum
    mask[..., 1] = 255

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    cv2.imshow("dense optical flow", rgb)
    cv2.imshow("OG image", img_new)
    cv2.waitKey(0)
    '''
    return flow

def show_nd_array(m:np.ndarray):
    m = Image.fromarray(m)
    m.show()

def main():
    print("ARGH!")
    n = 3
    i1 = get_image(n=3, t=300)
    i2 = get_image(n=3, t=690)
    ni1, ni2 = trnsformed_images(i1, i2)

    ni1 = Image.fromarray(ni1)
    ni2 = Image.fromarray(ni2)
    i1 = Image.fromarray(i1)
    i2 = Image.fromarray(i2)
    print("reconstructed")
    #i1.show(title="i1")
    i2.show(title="i2")
    # ni1.show(title="ni1")
    # ni2.show(title="ni1")


if __name__ == "__main__":
    main()
