using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Neuron
    {
        private List<Gene> inputs = new List<Gene>();
        private float value = 0;

        public float Value
        {
            get { return value; }
            set { this.value = value; }
        }

        public List<Gene> Inputs
        {
            get { return inputs; }
        }
    }
}
