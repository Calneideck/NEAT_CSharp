﻿using System;
using System.IO;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public static class Pool
    {
        private static List<Species> species = new List<Species>();
        private static int population;
        private static int outputs = 0;
        private static int inputs = 0;
        private static int innovation = 0;
        private static int generation = 1;
        private static int currentSpecies = 0;
        private static int currentGenome = 0;
        private static float maxFitness = 0;

        public static Random rnd = new Random();

        public const int StaleSpecies = 15;

        public static void Setup(int population, int inputs, int outputs)
        {
            Pool.population = population;
            Pool.inputs = inputs;
            Pool.outputs = outputs;
            innovation = outputs;

            for (int i = 0; i < population; i++)
            {
                Genome genome = Genome.BasicGenome();
                AddToSpecies(genome);
            }
        }

        public static void NewGeneration()
        {
            CullSpecies(false);
            RankGlobally();
            RemoveStaleSpecies();
            RankGlobally();

            RemoveWeakSpecies();
            float total = TotalAverageFitness();
            List<Genome> children = new List<Genome>();

            foreach (Species s in species)
            {
                int breed = (int)Math.Floor(s.AverageFitness / total * population) - 1;
                for (int i = 0; i < breed; i++)
                    children.Add(s.BreedChild());
            }

            CullSpecies(true);
            while (children.Count + species.Count < population)
            {
                Species s = species[rnd.Next(species.Count)];
                children.Add(s.BreedChild());
            }

            foreach (Genome child in children)
                AddToSpecies(child);

            generation++;
        }

        public static float[] Evaluate(float[] inputs)
        {
            return species[currentSpecies].Genomes[currentGenome].Evaluate(inputs);
        }

        /// <summary>
        ///     Returns true if it results in a new generation
        /// </summary>
        public static bool NextGenome()
        {
            currentGenome++;
            if (currentGenome >= species[currentSpecies].Genomes.Count)
            {
                currentGenome = 0;
                currentSpecies++;
                if (currentSpecies >= species.Count)
                {
                    currentSpecies = 0;
                    NewGeneration();
                    return true;
                }
            }

            return false;
        }

        public static void SetFitness(float fitness)
        {
            species[currentSpecies].Genomes[currentGenome].Fitness = fitness;
            if (fitness > maxFitness)
                maxFitness = fitness;
        }

        public static bool FitnessAlreadyMeasured()
        {
            return species[currentSpecies].Genomes[currentGenome].Fitness != 0;
        }

        public static void RankGlobally()
        {
            List<Genome> allGenomes = new List<Genome>();
            foreach (Species s in species)
                foreach (Genome g in s.Genomes)
                    allGenomes.Add(g);

            allGenomes.Sort((g1, g2) => g1.Fitness.CompareTo(g2.Fitness));
            for (int i = 0; i < allGenomes.Count; i++)
                allGenomes[i].GlobalRank = i;
        }

        public static float TotalAverageFitness()
        {
            float total = 0;
            foreach (Species s in species)
                total += s.AverageFitness;

            return total;
        }

        public static void CullSpecies(bool cullToOne)
        {
            foreach (Species s in species)
            {
                s.Genomes.Sort((g1, g2) => g1.Fitness.CompareTo(g2.Fitness));
                int remaining = cullToOne ? 1 : (int)Math.Ceiling(s.Genomes.Count * 0.5f);
                while (s.Genomes.Count > remaining)
                    s.Genomes.RemoveAt(0);
            }
        }

        public static void RemoveStaleSpecies()
        {
            List<Species> survived = new List<Species>();
            foreach (Species s in species)
            {
                s.Genomes.Sort((g1, g2) => -g1.Fitness.CompareTo(g2.Fitness));
                if (s.Genomes[0].Fitness > s.TopFitness)
                {
                    s.TopFitness = s.Genomes[0].Fitness;
                    s.Staleness = 0;
                }
                else
                    s.Staleness++;

                if (s.Staleness < StaleSpecies || s.TopFitness >= maxFitness)
                    survived.Add(s);
            }

            species.Clear();
            species.AddRange(survived);
        }

        public static void RemoveWeakSpecies()
        {
            List<Species> survived = new List<Species>();
            float total = TotalAverageFitness();
            foreach (Species s in species)
            {
                float breed = (float)Math.Floor(s.AverageFitness / total * population);
                if (breed >= 1)
                    survived.Add(s);
            }

            species.Clear();
            species.AddRange(survived);
        }

        public static void AddToSpecies(Genome child)
        {
            bool found = false;
            foreach (Species s in species)
                if (Genome.SameSpecies(child, s.Genomes[0]))
                {
                    s.Genomes.Add(child);
                    found = true;
                    break;
                }

            if (!found)
            {
                Species s = new Species();
                s.Genomes.Add(child);
                species.Add(s);
            }
        }

        public static int NewInnovation()
        {
            return ++innovation;
        }

        public static float Sigmoid(float x)
        {
            return 2 / (1 + (float)Math.Exp(-4.9f * x)) - 1;
        }

        public static void Save()
        {
            BinaryWriter writer = new BinaryWriter(File.Open("pool.dat", FileMode.Create));
            writer.Write(generation);
            writer.Write(maxFitness);
            writer.Write(innovation);
            writer.Write(species.Count);
            foreach (Species s in species)
            {
                writer.Write(s.TopFitness);
                writer.Write(s.Staleness);
                writer.Write(s.Genomes.Count);
                foreach (Genome g in s.Genomes)
                {
                    writer.Write(g.MaxNeuron);
                    foreach (KeyValuePair<string, float> kv in g.MutationRates)
                    {
                        writer.Write(kv.Key);
                        writer.Write(kv.Value);
                    }
                    writer.Write(g.Genes.Count);
                    foreach (Gene gene in g.Genes)
                    {
                        writer.Write(gene.input);
                        writer.Write(gene.output);
                        writer.Write(gene.weight);
                        writer.Write(gene.innovation);
                        writer.Write(gene.enabled);
                    }
                }
            }
            writer.Close();
        }

        public static void Load()
        {
            if (!File.Exists("pool.dat"))
                return;

            BinaryReader reader = new BinaryReader(File.Open("pool.dat", FileMode.Open));
            species.Clear();
            generation = reader.ReadInt32();
            maxFitness = reader.ReadSingle();
            innovation = reader.ReadInt32();
            int speciesCount = reader.ReadInt32();
            for (int i = 0; i < speciesCount; i++)
            {
                Species s = new Species();
                species.Add(s);
                s.TopFitness = reader.ReadSingle();
                s.Staleness = reader.ReadInt32();
                int genomeCount = reader.ReadInt32();
                for (int j = 0; j < genomeCount; j++)
                {
                    Genome g = new Genome();
                    s.Genomes.Add(g);
                    g.MaxNeuron = reader.ReadInt32();
                    for (int k = 0; k < 7; k++)
                        g.MutationRates[reader.ReadString()] = reader.ReadSingle();

                    int geneCount = reader.ReadInt32();
                    for (int l = 0; l < geneCount; l++)
                    {
                        Gene gene = new Gene();
                        g.Genes.Add(gene);
                        gene.input = reader.ReadInt32();
                        gene.output = reader.ReadInt32();
                        gene.weight = reader.ReadSingle();
                        gene.innovation = reader.ReadInt32();
                        gene.enabled = reader.ReadBoolean();
                    }
                    g.GenerateNetwork();
                }
            }
            reader.Close();
        }

        public static int Inputs
        {
            get { return inputs; }
        }

        public static int Outputs
        {
            get { return outputs; }
        }

        public static List<Species> Species
        {
            get { return species; }
        }

        public static int Generation
        {
            get { return generation; }
        }

        public static float MaxFitness
        {
            get { return maxFitness; }
            set { maxFitness = value; }
        }
    }
}
