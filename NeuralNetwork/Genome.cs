using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Genome
    {
        const float DeltaDisjoint = 2;
        const float DeltaWeights = 0.4f;
        const float DeltaThreshold = 1;

        const int StaleSpecies = 15;
        const float PerturbChance = 0.90f;

        const float MutateConnectionsChance = 0.25f;
        const float LinkMutationChance = 2.0f;
        const float BiasMutationChance = 0.40f;
        const float NodeMutationChance = 0.50f;
        const float EnableMutationChance = 0.2f;
        const float DisableMutationChance = 0.4f;
        const float StepSize = 0.1f;

        private float fitness = 0;
        private int maxNeuron = 0;
        private int globalRank = 0;
        private Dictionary<string, float> mutationRates = new Dictionary<string, float>()
        {
            { "connections", MutateConnectionsChance },
            { "link", LinkMutationChance },
            { "bias", BiasMutationChance },
            { "node", NodeMutationChance },
            { "enable", EnableMutationChance },
            { "disable", DisableMutationChance },
            { "step", StepSize },
        };

        private Dictionary<int, Neuron> neurons = new Dictionary<int, Neuron>();
        private List<Gene> genes = new List<Gene>();

        public Genome() { }

        public Genome(Genome genome)
        {
            maxNeuron = genome.maxNeuron;
            foreach (Gene gene in genome.genes)
                genes.Add(new Gene(gene));

            foreach (KeyValuePair<string, float> kv in genome.mutationRates)
                mutationRates[kv.Key] = kv.Value;
        }

        public void GenerateNetwork()
        {
            neurons.Clear();

            for (int i = 0; i < Pool.Inputs; i++)
                neurons.Add(i, new Neuron());

            for (int i = 0; i < Pool.Outputs; i++)
                neurons.Add(Pool.Inputs + i, new Neuron());

            genes.Sort((g1, g2) => g1.output.CompareTo(g2.output));
            foreach (Gene gene in genes)
            {
                if (gene.enabled)
                {
                    if (!neurons.ContainsKey(gene.output))
                        neurons.Add(gene.output, new Neuron());

                    Neuron neuron = neurons[gene.output];
                    neuron.Inputs.Add(gene);
                    if (!neurons.ContainsKey(gene.input))
                        neurons.Add(gene.input, new Neuron());
                }
            }
        }

        public float[] Evaluate(float[] inputs)
        {
            if (inputs.Length != Pool.Inputs)
            {
                Console.WriteLine("Incorrect number of inputs.");
                return null;
            }

            for (int i = 0; i < inputs.Length; i++)
                neurons[i].Value = inputs[i];

            // Hidden layer
            foreach (KeyValuePair<int, Neuron> kv in neurons)
            {
                if (kv.Key < Pool.Inputs + Pool.Outputs)
                    continue;

                float sum = 0;
                foreach (Gene gene in kv.Value.Inputs)
                {
                    Neuron other = neurons[gene.input];
                    sum += gene.weight * other.Value;
                }

                if (kv.Value.Inputs.Count > 0)
                    kv.Value.Value = Pool.Sigmoid(sum);
            }

            // Outputs
            foreach (KeyValuePair<int, Neuron> kv in neurons)
            {
                if (kv.Key < Pool.Inputs || kv.Key >= Pool.Inputs + Pool.Outputs)
                    continue;

                float sum = 0;
                foreach (Gene gene in kv.Value.Inputs)
                {
                    Neuron other = neurons[gene.input];
                    sum += gene.weight * other.Value;
                }

                if (kv.Value.Inputs.Count > 0)
                    kv.Value.Value = Pool.Sigmoid(sum);
            }

            float[] outputs = new float[Pool.Outputs];
            for (int i = 0; i < outputs.Length; i++)
                outputs[i] = neurons[Pool.Inputs + i].Value;

            return outputs;
        }

        public int RandomNeuron(bool canBeInput)
        {
            List<int> availNeurons = new List<int>();

            if (canBeInput)
                for (int i = 0; i < Pool.Inputs; ++i)
                    availNeurons.Add(i);

            for (int i = 0; i < Pool.Outputs; i++)
                availNeurons.Add(Pool.Inputs + i);

            foreach (Gene gene in genes)
            {
                if (!availNeurons.Contains(gene.input) && gene.input >= Pool.Inputs + Pool.Outputs)
                    availNeurons.Add(gene.input);

                if (!availNeurons.Contains(gene.output) && gene.output >= Pool.Inputs + Pool.Outputs)
                    availNeurons.Add(gene.output);
            }

            return availNeurons[Pool.rnd.Next(availNeurons.Count)];
        }

        public bool ContainsLink(Gene link)
        {
            foreach (Gene gene in genes)
                if (gene.input == link.input && gene.output == link.output)
                    return true;

            return false;
        }

        #region Mutations
        public void PointMutate()
        {
            float step = mutationRates["step"];

            foreach (Gene gene in genes)
                if (Pool.rnd.NextDouble() < PerturbChance)
                    gene.weight += (float)Pool.rnd.NextDouble() * step * 2 - step;
                else
                    gene.weight = (float)Pool.rnd.NextDouble() * 4 - 2;
        }

        public void LinkMutate(bool forceBias)
        {
            int n1 = RandomNeuron(true);
            int n2 = RandomNeuron(false);

            if (n1 < Pool.Inputs && n2 < Pool.Inputs)
                return;

            if (n2 < Pool.Inputs)
            {
                int temp = n1;
                n1 = n2;
                n2 = temp;
            }

            Gene newGene = new Gene();
            newGene.input = n1;
            newGene.output = n2;
            if (forceBias)
                newGene.input = Pool.Inputs - 1;

            if (ContainsLink(newGene))
                return;

            newGene.innovation = Pool.NewInnovation();
            newGene.weight = (float)Pool.rnd.NextDouble() * 4 - 2;
            genes.Add(newGene);
        }

        public void NodeMutate()
        {
            if (genes.Count == 0)
                return;

            Gene gene = genes[Pool.rnd.Next(genes.Count)];
            if (!gene.enabled)
                return;

            maxNeuron++;

            gene.enabled = false;
            Gene g1 = new Gene(gene);
            g1.output = maxNeuron;
            g1.weight = 1;
            g1.innovation = Pool.NewInnovation();
            g1.enabled = true;
            genes.Add(g1);

            Gene g2 = new Gene(gene);
            g2.input = maxNeuron;
            g2.innovation = Pool.NewInnovation();
            g2.enabled = true;
            genes.Add(g2);
        }

        public void EnableDisableMutate(bool enable)
        {
            List<Gene> availGenes = new List<Gene>();
            foreach (Gene gene in genes)
                if (gene.enabled != enable)
                    availGenes.Add(gene);

            if (availGenes.Count > 0)
                availGenes[Pool.rnd.Next(availGenes.Count)].enabled = enable;
        }

        public void Mutate()
        {
            // Mutate mutation rates
            foreach (string key in new List<string>(mutationRates.Keys))
            if (Pool.rnd.Next(2) == 1)
                    mutationRates[key] *= 0.95f;
                else
                    mutationRates[key] *= 1.05f;

            // Mutate gene weights
            if (Pool.rnd.NextDouble() < mutationRates["connections"])
                PointMutate();

            // Mutate link
            float p = mutationRates["link"];
            while (p > 0)
            {
                if (Pool.rnd.NextDouble() < p)
                    LinkMutate(false);
                p -= 1;
            }

            // Mutate Bias
            p = mutationRates["bias"];
            while (p > 0)
            {
                if (Pool.rnd.NextDouble() < p)
                    LinkMutate(true);
                p -= 1;
            }

            // Mutate node
            p = mutationRates["node"];
            while (p > 0)
            {
                if (Pool.rnd.NextDouble() < p)
                    NodeMutate();
                p -= 1;
            }

            // Mutate enable
            p = mutationRates["enable"];
            while (p > 0)
            {
                if (Pool.rnd.NextDouble() < p)
                    EnableDisableMutate(true);
                p -= 1;
            }

            // Mutate disable
            p = mutationRates["disable"];
            while (p > 0)
            {
                if (Pool.rnd.NextDouble() < p)
                    EnableDisableMutate(false);
                p -= 1;
            }
        }
        #endregion

        public float Fitness
        {
            get { return fitness; }
            set { fitness = value; }
        }

        public int GlobalRank
        {
            get { return globalRank; }
            set { globalRank = value; }
        }

        public List<Gene> Genes
        {
            get { return genes; }
        }

        public Dictionary<int, Neuron> Neurons
        {
            get { return neurons; }
        }

        public static Genome BasicGenome()
        {
            Genome genome = new Genome();
            genome.maxNeuron = Pool.Inputs;
            genome.Mutate();
            genome.GenerateNetwork();
            return genome;
        }

        public static Genome Crossover(Genome g1, Genome g2)
        {
            // Make sure g1 is the fitter parent
            if (g2.fitness > g1.fitness)
            {
                Genome temp = g1;
                g1 = g2;
                g2 = temp;
            }

            Genome child = new Genome();
            // Setup dictionary of genes from g2 based on innovation number
            Dictionary<int, Gene> innovations2 = new Dictionary<int, Gene>();
            foreach (Gene gene in g2.genes)
                innovations2.Add(gene.innovation, gene);

            foreach (Gene gene1 in g1.genes)
            {
                Gene gene2 = innovations2[gene1.innovation];
                if (gene2 != null && Pool.rnd.Next(2) == 1 && gene2.enabled)
                    child.genes.Add(new Gene(gene2));
                else
                    child.genes.Add(new Gene(gene1));
            }

            child.maxNeuron = Math.Max(g1.maxNeuron, g2.maxNeuron);

            foreach (KeyValuePair<string, float> kv in g1.mutationRates)
                child.mutationRates[kv.Key] = kv.Value;

            return child;
        }

        public static float Disjoint(Genome g1, Genome g2)
        {
            List<int> innovations1 = new List<int>();
            foreach (Gene gene in g1.genes)
                innovations1.Add(gene.innovation);

            List<int> innovations2 = new List<int>();
            foreach (Gene gene in g2.genes)
                innovations1.Add(gene.innovation);

            int disjoint = 0;

            foreach (int innovation in innovations1)
                if (!innovations2.Contains(innovation))
                    disjoint++;

            foreach (int innovation in innovations2)
                if (!innovations1.Contains(innovation))
                    disjoint++;

            return disjoint / (float)Math.Max(g1.genes.Count, g2.genes.Count);
        }

        public static float Weights(Genome g1, Genome g2)
        {
            Dictionary<int, Gene> i2 = new Dictionary<int, Gene>();
            foreach (Gene gene in g2.genes)
                i2.Add(gene.innovation, gene);

            float sum = 0;
            float coincident = 0;
            foreach (Gene gene in g1.genes)
            {
                if (i2.ContainsKey(gene.innovation))
                {
                    Gene gene2 = i2[gene.innovation];
                    sum += Math.Abs(gene.weight - gene2.weight);
                    coincident++;
                }
            }

            return sum / coincident;
        }

        public static bool SameSpecies(Genome g1, Genome g2)
        {
            float dd = DeltaDisjoint * Disjoint(g1, g2);
            float dw = DeltaWeights * Weights(g1, g2);
            return dd + dw < DeltaThreshold;
        }
    }
}
