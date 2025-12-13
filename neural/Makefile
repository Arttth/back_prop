CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -pedantic

all: run

build:
	$(CXX) $(CXXFLAGS) -I. main.cpp -o nn_app

run: build
	./nn_app
