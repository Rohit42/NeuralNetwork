using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural
{
    class Layer
    {
        public Node[] nodes { get; set; }

        public Layer(int numberOfNodes, int count) {
            nodes = new Node[numberOfNodes];
            for (int i = 0; i < nodes.Length; i++) {
                Node n = new Node(count);
            }
        }
    }
}
