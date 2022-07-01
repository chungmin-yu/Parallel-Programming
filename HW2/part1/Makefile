CXX=gcc
CPP=pi.c
TARGET=pi.out
FLAGS=-fPIC -std=c99 -g3 -O3 -I. -mavx2 -march=native -Wall -Wextra -pedantic -Wshadow
THREAD=-lpthread
#gcc -fPIC -std=c99 -g3 -O3 -I. -mavx2 -march=native -Wall -Wextra -pedantic -Wshadow pi.c -o pi.out -lpthread

all: ${CPP}
	${CXX} ${FLAGS} ${CPP} -o ${TARGET} ${THREAD}

.PHONY: clean

clean:
	rm *.out
	

