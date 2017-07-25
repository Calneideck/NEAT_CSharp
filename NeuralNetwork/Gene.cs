using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Gene
    {
        public int input = 0;
        public int output = 0;
        public float weight = 0;
        public bool enabled = true;
        public int innovation = 0;

        public Gene() { }

        public Gene(Gene gene)
        {
            input = gene.input;
            output = gene.output;
            weight = gene.weight;
            enabled = gene.enabled;
            innovation = gene.innovation;
        }
    }
}
