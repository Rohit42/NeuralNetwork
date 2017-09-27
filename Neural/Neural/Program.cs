using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural
{
    class Program {


        static void Main(String[] args) {
             DataSet ds = new DataSet(@"C:\Users\Rohit\Documents\Visual Studio 2017\Projects\Neural\Neural\signal.dat");
             DataSet db = new DataSet(@"C:\Users\Rohit\Documents\Visual Studio 2017\Projects\Neural\Neural\background.dat");
             DataSet data = new DataSet(@"C:\Users\Rohit\Documents\Visual Studio 2017\Projects\Neural\Neural\decisionTreeData.dat");
                      
             normalize(ds, db, data);
             MyNetwork myNetwork = new MyNetwork(ds, db);
             myNetwork.train();

            using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"C:\Users\Rohit\Documents\Visual Studio 2017\Projects\DecisionTree\DecisionTree\DecisionTree\finaldata.csv"))
                for (int i = 0; i < data.Points.Count; i++)
                {
                    file.WriteLine(i + "," + myNetwork.RunPoint(data.Points[i]));

                }
            Console.WriteLine("DONE!!");
            Console.ReadKey();
        }
            
        public static void normalize(DataSet ds, DataSet db, DataSet data) {
            double[] maxs = findMaxs(ds, db, data);

            
            foreach (var p in ds.Points) {
                for (int i = 0; i < p.Variables.Count(); i++) {
                    double x = maxs[i];
                    p.Variables[i] = p.Variables[i]/x;
                }
            }
            foreach (var p in db.Points)
            {
                for (int i = 0; i < p.Variables.Count(); i++)
                {
                    p.Variables[i] /= maxs[i];
                }
            }
            foreach (var p in data.Points)
            {
                for (int i = 0; i < p.Variables.Count(); i++)
                {
                    p.Variables[i] /= maxs[i];
                }
            }
        }
        public static double[] findMaxs(DataSet ds, DataSet db, DataSet data)
        {

            double[] maximums = new double[8];
            for (int i = 0; i < maximums.Length; i++)
            {
                maximums[i] = ds.Points[0].Variables[i];
            }

            for (int i = 0; i < ds.Points.Count(); i++)
            {
                for (int j = 0; j < ds.Points[i].Variables.Count(); j++)
                {
                    double x = ds.Points[i].Variables[j];
                    if (x > maximums[j])
                    {
                        maximums[j] = x;
                    }
                }
            }

            for (int i = 0; i < db.Points.Count(); i++)
            {
                for (int j = 0; j < db.Points[i].Variables.Count(); j++)
                {
                    double x = db.Points[i].Variables[j];
                    if (x > maximums[j])
                    {
                        maximums[j] = x;
                    }
                }
            }

            for (int i = 0; i < data.Points.Count(); i++)
            {
                for (int j = 0; j < data.Points[i].Variables.Count(); j++)
                {
                    double x = data.Points[i].Variables[j];
                    if (x > maximums[j])
                    {
                        maximums[j] = x;
                    }
                }
            }
            return maximums;
        }
    }

}

