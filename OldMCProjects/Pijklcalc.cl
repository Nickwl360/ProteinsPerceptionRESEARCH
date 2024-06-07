#define M 31
#define MAX 40
#define PI 3.14159265358979323

double newcomb(double N, double L) {
    if (N == 0 || L == 0 || N == L) {
        return 0.0;
    } else {
        double comb = (N * log(N) - L * log(L) - (N - L) * log(N - L)) +
                      0.5 * log(2.0 * PI * N) - 0.5 * log(2.0 * PI * L) - 0.5 * log(2.0 * PI * (N - L));

        return comb;
    }
}
double calcCombs(double NA, double NB, double la, double lA, double lb, double lB,double halpha, double ha, double kaa,double kab){
    double e = (halpha * (la-M/2 + lb-M/2) + ha * (lA-NA/2 + lB - NB/2) + kaa * ((la*lA - M*NA/2) + (lb*lB-M*NB/2)) + kab * ((lb*lA-M*NA/2) + (la*lB-M*NB/2)));
    return newcomb(NA, lA) + newcomb(NB, lB) + e;
}

__kernel void compute_Pijkl(__global double* Pijkl, const double halpha, const double ha, const double kaa, const double kab)
{
     int i = get_global_id(0);   //ARRAYindexes:[na,nb,j,l] [i,k,j,l]
     int na = (i/(MAX*MAX*MAX));
     int nb = (i/(MAX*MAX)) % MAX;
     int j = (i/MAX) %MAX;
     int l = i %MAX;

     double loop3 = 0;   //a=lA, b=lB, c=la,d=lb
     for(int a=0; a<=na;a++)
     {
        double loop4 = 0;
        for(int b=0; b<=nb;b++)
        {
            int c = j - a;
            int d = l - b;

            if(0 <= c && c <= M && 0 <= d && d <= M)
            {
                loop4 +=  exp(calcCombs(na,nb,c,a,d,b,halpha,ha,kaa,kab));
            }
        }
        loop3+=loop4;
    }
    Pijkl[i] = loop3;
}


