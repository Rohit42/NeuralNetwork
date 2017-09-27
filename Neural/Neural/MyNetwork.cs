using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural
{
    class MyNetwork
    {
        private Layer layer1;
        private Layer layer2;
        private Layer finalLayer;
        private DataSet signal;
        private DataSet background;
        double count = 0;
        public MyNetwork(DataSet signal, DataSet background) {
            this.signal = signal;
            this.background = background;
        }

        public void train() {
            initialize();
            int iterations = 250;
            for (int i = 0; i < iterations; i++)
            {
                double x = runNetwork();
                updateNextRound();
                Console.WriteLine("Current Error: " + x);
            }
        }
        public void updateNextRound() {

            findNextRoundLayer(layer1);
            findNextRoundLayer(layer2);
            findNextRoundLayer(finalLayer);

            updateLayerToNextRound(layer1);
            updateLayerToNextRound(layer2);
            updateLayerToNextRound(finalLayer);


        }
        public void updateLayerToNextRound(Layer p) {
            for (int i = 0; i < p.nodes.Count(); i++) {
                p.nodes[i].weights = p.nodes[i].nextWeights;
                p.nodes[i].bias = p.nodes[i].nextBias;

            }
        }
        public void findNextRoundLayer(Layer p) {
            double increment = 0.0025;
            double scalar = 2E-6;
            for (int i = 0;i < p.nodes.Count(); i++) {
                double error = runNetwork();
                for (int j = 0; j < p.nodes[i].weights.Count(); j++)
                {
                    p.nodes[i].weights[j] = p.nodes[i].weights[j] + increment;
                    double error1 = runNetwork();
                    double gradient = (error1-error)/ increment;
                    //Console.WriteLine("e  " +error);
                    //Console.WriteLine("E1  " + error1);
                    //Console.WriteLine(gradient);
                    p.nodes[i].weights[j] = p.nodes[i].weights[j] - increment;
                    //Console.WriteLine("Before: " +p.nodes[i].weights[j]);
                    p.nodes[i].nextWeights[j] = p.nodes[i].weights[j] - gradient*scalar;
                    //Console.WriteLine("After: " + p.nodes[i].nextWeights[j]);

                }
                double errorBias = runNetwork();
                p.nodes[i].bias = p.nodes[i].bias + increment;
                double errorBias1 = runNetwork();
                double gradientBias = (errorBias1 - errorBias) / increment;
                p.nodes[i].bias = p.nodes[i].bias - increment;
                p.nodes[i].nextBias = p.nodes[i].bias - gradientBias*scalar;

            }
        }
        public double runNetwork() {
            double error = 0;
            foreach (DataPoint d in signal.Points) {
                error += Math.Pow((1 - RunPoint(d)), 2);
            }
            foreach (DataPoint d in background.Points) {
                error += Math.Pow((-1-RunPoint(d)), 2);
            }

            return error;
        }
        public void initialize() {
            int c = 8;
            layer1 = new Layer(3, c);
            layer2 = new Layer(3, layer1.nodes.Count());
            finalLayer = new Layer(1, layer2.nodes.Count());
            for (int i = 0; i < layer1.nodes.Count(); i++)
            {
                layer1.nodes[i] = new Node(8);
            }
            for (int i = 0; i < layer2.nodes.Count(); i++)
            {
                layer2.nodes[i] = new Node(layer1.nodes.Count());
            }
            for (int i = 0; i < finalLayer.nodes.Count(); i++)
            {
                finalLayer.nodes[i] = new Node(layer2.nodes.Count());
            }
        }
        public double RunPoint(DataPoint p) {
            count++;
            for (int i = 0; i < layer1.nodes.Count(); i++)
            {
                layer1.nodes[i].loadInputs(p);
                layer1.nodes[i].findOutput();
            }
            for (int i = 0; i < layer2.nodes.Count(); i++)
            {
                layer2.nodes[i].loadInputs(layer1);
                layer2.nodes[i].findOutput();
            }
            for (int i = 0; i < finalLayer.nodes.Count(); i++)
            {
                finalLayer.nodes[i].loadInputs(layer2);
                finalLayer.nodes[i].findOutput();
            }
            double j = finalLayer.nodes[0].output;
            if (j==1) {
                Console.WriteLine(j);
            }

            return j;

        }
    }
}
