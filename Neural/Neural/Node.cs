using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural
{
    class Node
    {
        public double[] inputs { get; set; }
        public double[] weights { get; set; }
        public double[] nextWeights { get; set; }
        public double nextBias;
        public double output { get; set; }       
        public double bias { get; set; }
        private static Random r = new Random();

        public Node(int count) {
            inputs = new double[count];
            weights = new double[count];
            nextWeights = new double[count];

            for (int i = 0; i < weights.Count(); i++) {
                weights[i] = r.NextDouble();
            }

        }
        public void loadInputs(Layer j) {
            for (int i = 0; i<j.nodes.Count(); i++) {
                inputs[i] = j.nodes[i].output;
            }
        }
        public void loadInputs(DataPoint d) {
            for (int i = 0; i < d.Variables.Count(); i++) {
                inputs[i] = d.Variables[i];
             }

        }
        public double findOutput() {
            output = 0;
            for (int i = 0; i < inputs.Length; i++) {
                output += inputs[i] * weights[i];
            }
            output += bias;

            return Math.Tanh(output/10);
        }

    }
}
