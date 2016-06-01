package Classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import org.knowm.xchart.Chart;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SeriesLineStyle;
import org.knowm.xchart.StyleManager.ChartType;
import static org.knowm.xchart.StyleManager.ChartType.Scatter;
import org.knowm.xchart.StyleManager.LegendPosition;
import org.knowm.xchart.SwingWrapper;
import weka.core.matrix.Matrix;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Jessa
 */
public class LogReg {

    int numIter = 1000;
    double alpha = 0.1;
    double lambda = 0;

    GradientDescentValues1 gradientDescent(Matrix X, Matrix y, Matrix theta, double alpha, int numIterations, double lambda) {

        CostFunctionValues1 CFV = new CostFunctionValues1();
        GradientDescentValues1 GDV = new GradientDescentValues1();

        int m = X.getRowDimension();
        Matrix J_history = new Matrix(numIterations, 1);
        for (int i = 0; i < numIterations; i++) {
            CFV = costFunction(theta, X, y);
            J_history.set(i, 0, CFV.getJ().get(0, 0));
            theta = theta.minus(CFV.getGrad().times(alpha));
        }
        GDV.setTheta(theta);
        GDV.setCostHistory(J_history);

        return GDV;
    }

    CostFunctionValues1 costFunction(Matrix theta, Matrix X, Matrix y) {
        int row = X.getRowDimension();
        LogReg lg = new LogReg();
        CostFunctionValues1 CFV = new CostFunctionValues1();

        Matrix J = new Matrix(1, 1);
        Matrix sigmoidValues = sigmoid(X.times(theta));

        Matrix ones = new Matrix(row, 1);
        ones.setMatrix(0, row - 1, 0, 0, new Matrix(row, 1, 1.0));
        double v = 1.d / row;
        Matrix Jtmp = y.uminus().arrayTimes(lg.log(sigmoidValues)).minus(ones.minus(y).
                arrayTimes(lg.log(ones.minus(sigmoidValues)))).times(v);

        //MatrixFunctions.log(h).print(8, 5);
        for (int i = 0; i < Jtmp.getRowDimension(); i++) {
            J.set(0, 0, J.get(0, 0) + Jtmp.get(i, 0));
        }

        Matrix gradTmp = X.transpose().times(sigmoidValues.minus(y)).times(v);

        //Regularizaiton
        Matrix t = theta.getMatrix(1, theta.getRowDimension() - 1, 0, 0);
        t = (Matrix) lg.pow(t, 2);
        t.times(lambda / 2 * row);
        Matrix sum = new Matrix(1, 1);

        for (int i = 0; i < t.getRowDimension(); i++) {
            sum.set(0, 0, sum.get(0, 0) + t.get(i, 0));
        }
        J.plusEquals(sum);

        t = theta.getMatrix(1, theta.getRowDimension() - 1, 0, 0);
        t.timesEquals(lambda / row);

        for (int i = 1; i < gradTmp.getRowDimension(); i++) {
            gradTmp.set(i, 0, gradTmp.get(i, 0) + t.get(i - 1, 0));
        }
        CFV.setGrad(gradTmp);
        CFV.setJ(J);
        return CFV;
    }

    public Matrix log(Matrix x) {
        Matrix tmp = new Matrix(x.getRowDimension(), x.getColumnDimension());
        for (int r = 0; r < x.getRowDimension(); r++) {
            tmp.set(r, 0, Math.log(x.get(r, 0)));
        }
        return tmp;
    }

    public static Matrix pow(Matrix X, int pow) {
        Matrix x = X.copy();

        for (int r = 0; r < x.getRowDimension(); r++) {
            x.set(r, 0, Math.pow(x.get(r, 0), pow));
        }
        return x;
    }

    Matrix sigmoid(Matrix z) {
        // g = 1.0 ./ (1.0 + exp(-z));
        Matrix m = z.copy();
        for (int i = 0; i < z.getRowDimension(); i++) {
            m.set(i, 0, 1.d / (1.d + Math.exp(-z.get(i, 0))));
        }
        return m;
    }

    Matrix predict(Matrix theta, Matrix X) {
        int row = X.getRowDimension();
        Matrix prob = new Matrix(row, 1);
        Matrix hyp = sigmoid(X.times(theta));
        for (int i = 0; i < row; i++) {
            if (hyp.get(i, 0) >= 0.5) {
                prob.set(i, 0, 1);
            } else {
                prob.set(i, 0, 0);
            }
        }
        return prob;

    }

    double accuracy(Matrix p, Matrix y) {
        double pred = 0.0;

        for (int i = 0; i < p.getRowDimension(); i++) {
            if (p.get(i, 0) == y.get(i, 0)) {
                pred++;
            }
        }
        double acc = pred / y.getRowDimension() * 100;
        return acc;
    }

    public void plot(Matrix X, Matrix Y) {
        Chart chart = new Chart(500, 500);
        chart.setXAxisTitle("Exam 1 Score");
        chart.setYAxisTitle("Exam 2 Score");
        ArrayList<Double> yesX = new ArrayList<>();
        ArrayList<Double> yesY = new ArrayList<>();
        ArrayList<Double> noX = new ArrayList<>();
        ArrayList<Double> noY = new ArrayList<>();

        for (int i = 0; i < X.getRowDimension(); i++) {
            if (Y.get(i, 0) == 1) {
                yesX.add(X.get(i, 0));
                yesY.add(X.get(i, 1));
            } else {
                noX.add(X.get(i, 0));
                noY.add(X.get(i, 1));
            }
        }
        System.out.println("");
        chart.addSeries("Admitted", yesX, yesY).setLineStyle(SeriesLineStyle.NONE);
        chart.addSeries("Not Admitted", noX, noY).setLineStyle(SeriesLineStyle.NONE);
        new SwingWrapper(chart).displayChart("Data Plot");
    }

    public static void main(String[] args) throws FileNotFoundException {

        LogReg lg = new LogReg();
        CostFunctionValues1 CFV = new CostFunctionValues1();
        GradientDescentValues1 GDV = new GradientDescentValues1();

        System.out.println("Loading data for Logistic Regression...");
        double[][] M = lg.loadData("ex2data1.txt");
        Matrix data = new Matrix(M);
        int row = data.getRowDimension();

        Matrix Xvalues = data.getMatrix(0, row - 1, 0, 1);//gets the features
        Matrix Yvalues = data.getMatrix(0, row - 1, 2, 2);//gets the output

        FeatureNormalizationValues1 norm = lg.featureNormalization(Xvalues);
        Matrix newX = new Matrix(row, 3);

        //add intercepts of 1.0
        newX.setMatrix(0, row - 1, 0, 0, new Matrix(row, 1, 1.0));
        newX.setMatrix(0, row - 1, 1, 2, norm.X);

        System.out.println("Normalized Features:");
        newX.print(8, 5);
        lg.plot(Xvalues, Yvalues);

        System.out.println("Cost at initial theta: ");
        // lg.sigmoid(newX);
        Matrix theta = new Matrix(newX.getColumnDimension(), 1);//automatically assigns vector of zeros to the matrix

        CFV = lg.costFunction(theta, newX, Yvalues);
        CFV.getJ().print(3, 5);
        System.out.println("Initial Gradient Descent: ");
        CFV.getGrad().print(3, 5);
        Matrix newTheta = new Matrix(theta.getRowDimension(), theta.getColumnDimension());
        GDV = lg.gradientDescent(newX, Yvalues, newTheta, lg.alpha, lg.numIter, lg.lambda);

        Chart chart = QuickChart.getChart("Cost History", "Iterations", "Cost Value", null, null, GDV.getCostHistory().getRowPackedCopy());
        new SwingWrapper(chart).displayChart();

        Matrix test = new Matrix(3, 1);
        test.set(0, 0, 1);
        test.set(1, 0, ((45 - norm.getMu().get(0, 0)) / norm.getSigma().get(0, 0)));
        test.set(2, 0, ((85 - norm.getMu().get(0, 1)) / norm.getSigma().get(0, 1)));

        Matrix prob = lg.sigmoid(GDV.getTheta().transpose().times(test));

        System.out.println("Students with Scores 45 and 85 Admission Probability");
        prob.print(8, 5);

        Matrix predict = lg.predict(GDV.getTheta(), newX);
        System.out.println("Train Accuracy: ");
        System.out.println(lg.accuracy(predict, Yvalues));

    }

    public FeatureNormalizationValues1 featureNormalization(Matrix X) {
        FeatureNormalizationValues1 FNV = new FeatureNormalizationValues1();
        int xC = X.getColumnDimension();
        int xR = X.getRowDimension();
        Matrix normX = X;
        Matrix mu = new Matrix(1, xC);
        Matrix sigma = new Matrix(1, xC);

        for (int row = 0; row < xR; row++) {
            for (int col = 0; col < xC; col++) {
                mu.set(0, col, mu.get(0, col) + normX.get(row, col));
                sigma.set(0, col, sigma.get(0, col) + Math.pow(normX.get(row, col), 2));
            }
        }
        for (int col = 0; col < xC; col++) {
            sigma.set(0, col, Math.sqrt(((sigma.get(0, col)) - (Math.pow(mu.get(0, col), 2) / normX.getRowDimension())) / (normX.getRowDimension() - 1)));
            mu.set(0, col, mu.get(0, col) / normX.getRowDimension());
        }

        for (int row = 0; row < xR; row++) {
            for (int col = 0; col < xC; col++) {
                normX.set(row, col, (normX.get(row, col) - mu.get(0, col)) / sigma.get(0, col));
            }
        }

        FNV.setMu(mu);
        FNV.setSigma(sigma);
        FNV.setX(X);

        return FNV;
    }

    public double[][] loadData(String file) throws FileNotFoundException {
        double[][] a = null;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(new File(file)));
            String st = "";
            a = new double[100][3];
            int i = 0, j = 0;
            while ((st = reader.readLine()) != null) {
                String[] s = st.split(",");
                for (String str : s) {
                    a[i][j++] = new Double(str).doubleValue();
                }
                j = 0;
                i++;
            }
            reader.close();
            System.out.println("Reading Successful");

        } catch (Exception e) {
            System.out.println("No Such File");
        }
        return a;

    }

    public static class CostFunctionValues1 {

        private Matrix J;
        private Matrix grad;
        // add getters and setters 

        /**
         * @return the J
         */
        public Matrix getJ() {
            return J;
        }

        /**
         * @param J the J to set
         */
        public void setJ(Matrix J) {
            this.J = J;
        }

        /**
         * @return the grad
         */
        public Matrix getGrad() {
            return grad;
        }

        /**
         * @param grad the grad to set
         */
        public void setGrad(Matrix grad) {
            this.grad = grad;
        }
    }

    static class GradientDescentValues1 {

        Matrix theta;
        Matrix costHistory;

        public Matrix getTheta() {
            return theta;
        }

        public void setTheta(Matrix theta) {
            this.theta = theta;
        }

        public Matrix getCostHistory() {
            return costHistory;
        }

        public void setCostHistory(Matrix costHistory) {
            this.costHistory = costHistory;
        }
    }

    class FeatureNormalizationValues1 {

        Matrix X;
        Matrix mu;
        Matrix sigma;

        public Matrix getX() {
            return X;
        }

        public void setX(Matrix X) {
            this.X = X;
        }

        public Matrix getMu() {
            return mu;
        }

        public void setMu(Matrix mu) {
            this.mu = mu;
        }

        public Matrix getSigma() {
            return sigma;
        }

        public void setSigma(Matrix sigma) {
            this.sigma = sigma;
        }

    }

}
