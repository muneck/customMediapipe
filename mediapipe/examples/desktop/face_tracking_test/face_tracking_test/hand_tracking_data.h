#ifndef HAND_TRACKING_DATA_H
#define HAND_TRACKING_DATA_H

struct PoseInfo {
	float x;
	float y;
	float z;
};

struct Gesture
{
	float turn;
	float tilt;
	float nod;
};

enum HandUp_HandDown
{
	NoHand = -1,
	HandUp = 1,
	HandDown = 2
};


#endif // !HAND_TRACKING_DATA_H
