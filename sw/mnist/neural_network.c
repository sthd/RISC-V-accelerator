#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "include/neural_network.h"

// Convert a pixel value from 0-255 to one from 0 to 1
#define PIXEL_SCALE(x) (((uint32_t) (x)<<8) )
//#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)

/**
 * Calculate the softmax vector from the activations. This uses a more
 * numerically stable algorithm that normalises the activations to prevent
 * large exponents.
 */
void neural_network_softmax(fix16_t * activations, int length)
{
    int i;
    fix16_t sum, max;
    fix16_t tmp;

    for (i = 1, max = activations[0]; i < length; i++) {
        tmp = fix16_ssub(activations[i],  max);
        if ((tmp & 0x80000000) ==0) {
        //if (activations[i] > max) {
            max = activations[i];
        }
    }

    for (i = 0, sum = 0; i < length; i++) {
	activations[i] = fix16_exp(fix16_ssub(activations[i], max));
        //activations[i] = exp(activations[i] - max);
        sum = fix16_sadd(sum, activations[i]);
	//sum += activations[i];
    }

    for (i = 0; i < length; i++) {
        activations[i] = fix16_div(activations[i], sum);
        //activations[i] /= sum;
    }
}

/**
 * Use the weights and bias vector to forward propogate through the neural
 * network and calculate the activations.
 */
void neural_network_hypothesis(uint8_t * image, neural_network_t * network, fix16_t activations[MNIST_LABELS])
{
    int i, j;
    fix16_t tmp;

    for (i = 0; i < MNIST_LABELS; i++) {
        activations[i] = network->b[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            tmp = fix16_smul(network->W[i][j], PIXEL_SCALE(image[j]));
            activations[i] = fix16_sadd(activations[i], tmp);
            //activations[i] += network->W[i][j] * PIXEL_SCALE(image[j]);
        }
    }

    neural_network_softmax(activations, MNIST_LABELS);
}

