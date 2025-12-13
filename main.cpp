#include <iostream>
#include "nw/NeuralNetwork.h"
#include "nw/LayerGen.h"


int main()
{
    nw::NeuralNetwork<double> nn(3);

    nn.init({
        nw::LayerGen<double>::gen(nw::Input, 10),
        nw::LayerGen<double>::gen(nw::ReLu, 10000),
        nw::LayerGen<double>::gen(nw::ReLu, 10),
    });

    nn.forward({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    nn.printOutput();


    std::cout << "nw study" << std::endl;

    

    return 0;
}
