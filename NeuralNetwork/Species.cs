using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Species
    {
        const float CrossoverChance = 0.75f;

        private float topFitness = 0;
        private float staleness = 0;
        private List<Genome> genomes = new List<Genome>();

        public List<Genome> Genomes
        {
            get { return genomes; }
        }

        public float AverageFitness
        {
            get
            {
                int avg = 0;
                foreach (Genome genome in genomes)
                    avg += genome.GlobalRank;

                return avg / (float)genomes.Count;
            }
        }

        public Genome BreedChild()
        {
            Genome child;
            Genome g1 = genomes[Pool.rnd.Next(genomes.Count)];

            if (Pool.rnd.NextDouble() < CrossoverChance)
            {
                Genome g2 = genomes[Pool.rnd.Next(genomes.Count)];
                child = Genome.Crossover(g1, g2);
            }
            else
                child = new Genome(g1);

            child.Mutate();
            child.GenerateNetwork();
            return child;
        }

        public float TopFitness
        {
            get { return topFitness; }
            set { topFitness = value; }
        }

        public float Staleness
        {
            get { return staleness; }
            set { staleness = value; }
        }
    }
}
