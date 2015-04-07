all:	red_object_tracking

red_object_tracking: semaphore.cpp
	g++ semaphore.cpp -o semaphore `pkg-config opencv cvblob --cflags --libs`
