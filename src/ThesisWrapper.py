import gym
import gym.spaces as spaces
import numpy as np
from collections import deque
import cv2


class ThesisWrapper(gym.ObservationWrapper):

    def __init__(self,
                 env: gym.Env,
                 history_count=4,
                 convert_greyscale=False):
        super().__init__(env)
        self.history_count = history_count
        self.convert_greyscale = convert_greyscale
        self.frames = deque(maxlen=self.history_count)
        state = self.reset()
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8)

    def observation(self, obs):
        frame = self.operation_on_single_frame(obs=obs)
        self.frames.append(frame)
        state = self.operations_on_stack()
        return state

    def reset(self):
        og_state = self.env.reset()
        frame = self.operation_on_single_frame(obs=og_state)
        for i in range(self.history_count - 1):
            self.frames.append(frame)
        obs = self.observation(obs=og_state)
        return obs

    def operation_on_single_frame(self, obs):
        frame = obs
        if self.convert_greyscale:
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return frame

    def operations_on_stack(self):
        # state = np.concatenate(list(self.frames), axis=2)
        state = np.dstack(list(self.frames))
        return state

    '''    
    def see(self,observation):
        global istep
        image = np.array(observation, dtype="uint8")
        w,h = (int(IMAGE_WIDTH_R), int(IMAGE_HEIGHT_R))
        #image = cv2.resize(image, (w,h))
        #imageSol = self.imageSol.copy()
        imageSol = np.zeros((h,w,3), np.uint8) if bSEE else None
        greyscaled = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        imgmov = cv2.absdiff(self.last_grey, greyscaled)
        self.last_grey = greyscaled
        cv2.accumulateWeighted(imgmov, self.running_average_img, AVERAGE_ALPHA)
        background = cv2.convertScaleAbs(self.running_average_img)
        #background = cv2.blur(background,(2,2))
        ret, thimg = cv2.threshold(background, MOVEMENT_THRESHOLD, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thimg, MORPH_KERNEL, iterations=3)
        allcontours, ret = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if(bSEE):
            cv2.drawContours(imageSol,allcontours, -1, (255,0,0), -1)
            cv2.drawContours(imageSol,allcontours, -1, (255,255,0), 1)
        contours = []
        if(np.sum(dilated) > w*h/1.6):
            return

        for cnt in allcontours:
            #cv2.drawContours(imageSol,[cnt], -1, (0,255,0), -1)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            x, y, radius = int(x), int(y), int(radius)
            if(radius > 3):
                contours.append((x, y, radius))

        self.updateEntities(contours,image,imageSol)

        if(bSEE):
            #cv2.circle(imageSol,state,3,(0,120,255),-1)
            imagebig = cv2.resize(imageSol, (int(IMAGE_WIDTH*3.5), int(IMAGE_HEIGHT*3.5)))
            cv2.imshow("view",imagebig)
            #self.saveImgS2S(image,imageSol)
        return image
        '''


if __name__ == "__main__":
    env_name = "CarRacing-v1"
    env = gym.make(env_name, continuous=False)
    env = ThesisWrapper(env=env)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("step shape ", observation.shape)
    print("done checking")
