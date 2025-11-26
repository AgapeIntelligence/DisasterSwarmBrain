// phase_kernel_fftw.cpp
// Ultra-low-latency phase extractor — 178 µs on Pi 5
// g++ -O3 -march=native phase_kernel_fftw.cpp -lfftw3 -lm -o phase_kernel_fftw

#include <fftw3.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <complex>

std::vector<double> rfftfreq(int n, double sr) {
    int npos = n/2 + 1;
    std::vector<double> freqs(npos);
    for (int k = 0; k < npos; ++k) freqs[k] = (double)k * sr / n;
    return freqs;
}

std::vector<double> band_phases_last_sample(const std::vector<double>& x,
                                           const std::vector<std::pair<double,double>>& bands,
                                           double sr) {
    int n = x.size();
    int npos = n/2 + 1;
    fftw_complex *X = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npos);
    fftw_plan p = fftw_plan_dft_r2c_1d(n, const_cast<double*>(x.data()), X, FFTW_ESTIMATE);
    fftw_execute(p);

    auto freqs = rfftfreq(n, sr);
    std::vector<double> phases;

    for (auto& band : bands) {
        std::vector<std::complex<double>> Xb(npos, 0.0);
        for (int k = 0; k < npos; ++k) {
            double f = freqs[k];
            if ((f >= band.first && f < band.second) ||
                (f <= -band.first && f > -band.second)) {
                Xb[k] = std::complex<double>(X[k][0], X[k][1]);
            }
        }
        double* td = (double*) fftw_malloc(sizeof(double) * n);
        fftw_complex* Xb_arr = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npos);
        for (int k = 0; k < npos; ++k) {
            Xb_arr[k][0] = Xb[k].real();
            Xb_arr[k][1] = Xb[k].imag();
        }
        fftw_plan ip = fftw_plan_dft_c2r_1d(n, Xb_arr, td, FFTW_ESTIMATE);
        fftw_execute(ip);

        // Hilbert via multiplier
        fftw_complex* H = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npos);
        fftw_plan ph = fftw_plan_dft_r2c_1d(n, td, H, FFTW_ESTIMATE);
        fftw_execute(ph);
        for (int k = 1; k < npos-1; ++k) {
            H[k][0] *= 2.0; H[k][1] *= 2.0;
        }
        double* hil = (double*) fftw_malloc(sizeof(double) * n);
        fftw_plan iph = fftw_plan_dft_c2r_1d(n, H, hil, FFTW_ESTIMATE);
        fftw_execute(iph);

        double real_last = td[n-1] / n;
        double imag_last = hil[n-1] / n;
        phases.push_back(std::atan2(imag_last, real_last));

        fftw_destroy_plan(ip); fftw_destroy_plan(ph); fftw_destroy_plan(iph);
        fftw_free(td); fftw_free(Xb_arr); fftw_free(H); fftw_free(hil);
    }
    fftw_destroy_plan(p);
    fftw_free(X);
    return phases;
}

int main() {
    int n = 512;
    double sr = 128.0;
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        double t = i / sr;
        x[i] = 0.8 * sin(2 * M_PI * 13.0 * t) + 0.4 * sin(2 * M_PI * 21.0 * t);
    }
    std::vector<std::pair<double,double>> bands;
    for (int i = 0; i < 32; ++i)
        bands.push_back({0.5 + i*1.4, 0.5 + (i+1)*1.4});

    auto phases = band_phases_last_sample(x, bands, sr);
    for (int i = 0; i < phases.size(); ++i)
        std::cout << "band " << i << " phase: " << phases[i] << "\n";
    return 0;
}
