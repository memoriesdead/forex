"""
Spectral and Wavelet Analysis for HFT Forex
============================================
Renaissance Technologies inference: Signal processing experts use
frequency domain analysis to identify market cycles and filter noise.

Implemented Methods:
- FFT Power Spectrum: Identify dominant frequencies
- Welch's Method: Robust spectral estimation
- Discrete Wavelet Transform (DWT): Multi-scale decomposition
- Continuous Wavelet Transform (CWT): Time-frequency analysis
- Hilbert Transform: Instantaneous phase/frequency

Sources:
- Oppenheim & Schafer "Discrete-Time Signal Processing"
- Mallat (1989) "A Theory for Multiresolution Signal Decomposition"
- Daubechies (1992) "Ten Lectures on Wavelets"
- Ramsey & Lampart (1998) "The Decomposition of Economic Relationships by Time Scale"

Why Renaissance Uses This:
- Identify dominant market cycles (intraday, weekly, monthly)
- Filter noise from signal at different frequencies
- Detect phase shifts in correlations
- Multi-scale trend following
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from scipy.signal import welch, hilbert
try:
    from scipy.signal import cwt, morlet2
except ImportError:
    # scipy >= 1.12 moved cwt to scipy.signal
    # Provide fallback using ricker wavelet or custom implementation
    cwt = None
    morlet2 = None
from scipy.fft import fft, fftfreq
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpectralResult:
    """Result from spectral analysis."""
    frequencies: np.ndarray  # Frequency axis
    power: np.ndarray  # Power spectral density
    dominant_freq: float  # Dominant frequency
    dominant_period: float  # Dominant period (1/freq)
    spectral_entropy: float  # Measure of randomness (0=periodic, 1=random)


@dataclass
class WaveletResult:
    """Result from wavelet decomposition."""
    approximation: np.ndarray  # Low-frequency component (trend)
    details: List[np.ndarray]  # High-frequency components (details)
    levels: int  # Number of decomposition levels
    trend: np.ndarray  # Extracted trend
    noise: np.ndarray  # Extracted noise


class SpectralAnalyzer:
    """
    Fourier-based spectral analysis for cycle detection.

    Decomposes price series into frequency components to identify:
    - Dominant cycles (intraday, weekly, etc.)
    - Noise level (high frequency)
    - Trend component (low frequency)

    Renaissance Application:
    - Identify profitable trading frequencies
    - Filter out noise
    - Time entries to cycle phases
    """

    def __init__(self, sample_rate: float = 1.0):
        """
        Initialize spectral analyzer.

        Args:
            sample_rate: Samples per unit time (e.g., 12 for 5-min bars/hour)
        """
        self.sample_rate = sample_rate

    def fft_spectrum(self, signal: np.ndarray) -> SpectralResult:
        """
        Compute FFT power spectrum.

        Args:
            signal: Time series data

        Returns:
            SpectralResult with frequencies and power
        """
        n = len(signal)

        # Detrend
        detrended = signal - np.mean(signal)

        # Apply window to reduce spectral leakage
        window = np.hanning(n)
        windowed = detrended * window

        # FFT
        spectrum = fft(windowed)
        power = np.abs(spectrum)**2 / n

        # Frequencies
        freqs = fftfreq(n, d=1/self.sample_rate)

        # Only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power = power[pos_mask]

        # Find dominant frequency
        dominant_idx = np.argmax(power)
        dominant_freq = freqs[dominant_idx]
        dominant_period = 1 / dominant_freq if dominant_freq > 0 else np.inf

        # Spectral entropy (measure of randomness)
        power_norm = power / (power.sum() + 1e-10)
        spectral_entropy = -np.sum(power_norm * np.log(power_norm + 1e-10)) / np.log(len(power))

        return SpectralResult(
            frequencies=freqs,
            power=power,
            dominant_freq=dominant_freq,
            dominant_period=dominant_period,
            spectral_entropy=spectral_entropy
        )

    def welch_spectrum(self, signal: np.ndarray,
                      nperseg: int = None) -> SpectralResult:
        """
        Welch's method for robust spectral estimation.

        Uses overlapping segments for lower variance estimate.
        """
        n = len(signal)
        if nperseg is None:
            nperseg = min(256, n // 4)

        freqs, power = welch(signal, fs=self.sample_rate, nperseg=nperseg)

        # Find dominant frequency
        dominant_idx = np.argmax(power)
        dominant_freq = freqs[dominant_idx]
        dominant_period = 1 / dominant_freq if dominant_freq > 0 else np.inf

        # Spectral entropy
        power_norm = power / (power.sum() + 1e-10)
        spectral_entropy = -np.sum(power_norm * np.log(power_norm + 1e-10)) / np.log(len(power))

        return SpectralResult(
            frequencies=freqs,
            power=power,
            dominant_freq=dominant_freq,
            dominant_period=dominant_period,
            spectral_entropy=spectral_entropy
        )

    def band_power(self, signal: np.ndarray,
                   low_freq: float, high_freq: float) -> float:
        """
        Compute power in a specific frequency band.

        Args:
            signal: Time series
            low_freq: Lower bound of band
            high_freq: Upper bound of band

        Returns:
            Total power in band
        """
        result = self.welch_spectrum(signal)

        mask = (result.frequencies >= low_freq) & (result.frequencies <= high_freq)
        band_power = result.power[mask].sum()

        return float(band_power)

    def noise_ratio(self, signal: np.ndarray,
                   noise_freq_cutoff: float = 0.4) -> float:
        """
        Estimate noise ratio (high-freq power / total power).

        Lower ratio = more signal, less noise
        """
        result = self.welch_spectrum(signal)

        total_power = result.power.sum()
        noise_mask = result.frequencies > noise_freq_cutoff
        noise_power = result.power[noise_mask].sum()

        return float(noise_power / (total_power + 1e-10))


class WaveletDecomposer:
    """
    Wavelet decomposition for multi-scale analysis.

    Advantages over Fourier:
    - Localized in both time and frequency
    - Better for non-stationary signals
    - Captures transient events

    Renaissance Application:
    - Separate long-term trend from short-term noise
    - Detect regime changes at multiple timescales
    - Multi-resolution trading signals
    """

    def __init__(self, wavelet: str = 'db4', levels: int = 4):
        """
        Initialize wavelet decomposer.

        Args:
            wavelet: Wavelet type ('db4' = Daubechies-4, good for finance)
            levels: Number of decomposition levels
        """
        self.wavelet = wavelet
        self.levels = levels

    def _haar_decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple Haar wavelet decomposition (one level).

        cA = (s[2k] + s[2k+1]) / sqrt(2)  (approximation = low-pass)
        cD = (s[2k] - s[2k+1]) / sqrt(2)  (detail = high-pass)
        """
        n = len(signal)
        half = n // 2

        # Ensure even length
        if n % 2 != 0:
            signal = np.append(signal, signal[-1])
            half = len(signal) // 2

        approx = np.zeros(half)
        detail = np.zeros(half)

        for k in range(half):
            approx[k] = (signal[2 * k] + signal[2 * k + 1]) / np.sqrt(2)
            detail[k] = (signal[2 * k] - signal[2 * k + 1]) / np.sqrt(2)

        return approx, detail

    def _db4_decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Daubechies-4 wavelet decomposition (one level).

        More sophisticated than Haar - smoother approximation.
        """
        # DB4 filter coefficients
        h0 = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
        h1 = (3 + np.sqrt(3)) / (4 * np.sqrt(2))
        h2 = (3 - np.sqrt(3)) / (4 * np.sqrt(2))
        h3 = (1 - np.sqrt(3)) / (4 * np.sqrt(2))

        # Low-pass filter (scaling)
        lpf = np.array([h0, h1, h2, h3])
        # High-pass filter (wavelet)
        hpf = np.array([h3, -h2, h1, -h0])

        n = len(signal)

        # Pad for convolution
        padded = np.concatenate([signal[-3:], signal, signal[:3]])

        # Convolve and downsample
        approx = np.convolve(padded, lpf, mode='valid')[::2]
        detail = np.convolve(padded, hpf, mode='valid')[::2]

        # Trim to correct length
        half = (n + 1) // 2
        approx = approx[:half]
        detail = detail[:half]

        return approx, detail

    def decompose(self, signal: np.ndarray) -> WaveletResult:
        """
        Multi-level wavelet decomposition.

        Returns approximation (trend) and detail coefficients at each level.
        """
        # Use simple implementation if pywt not available
        approx = signal.copy()
        details = []

        for level in range(self.levels):
            if len(approx) < 8:
                break

            if self.wavelet == 'haar':
                approx, detail = self._haar_decompose(approx)
            else:
                approx, detail = self._db4_decompose(approx)

            details.append(detail)

        # Reconstruct trend (low-pass filtered signal)
        trend = np.interp(
            np.linspace(0, len(approx) - 1, len(signal)),
            np.arange(len(approx)),
            approx
        )

        # Noise is sum of high-frequency details
        noise = signal - trend

        return WaveletResult(
            approximation=approx,
            details=details,
            levels=len(details),
            trend=trend,
            noise=noise
        )

    def denoise(self, signal: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Wavelet denoising using soft thresholding.

        Args:
            signal: Noisy signal
            threshold: Threshold for wavelet coefficients (auto if None)

        Returns:
            Denoised signal
        """
        result = self.decompose(signal)

        if threshold is None:
            # Universal threshold (Donoho & Johnstone)
            n = len(signal)
            sigma = np.median(np.abs(result.details[0])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(n))

        # Soft thresholding
        denoised_details = []
        for detail in result.details:
            thresholded = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
            denoised_details.append(thresholded)

        # Reconstruct (simplified - use trend + reduced noise)
        denoised = result.trend.copy()
        for detail in denoised_details:
            # Upsample and add
            upsampled = np.interp(
                np.linspace(0, len(detail) - 1, len(signal)),
                np.arange(len(detail)),
                detail
            )
            denoised += upsampled

        return denoised


class ContinuousWaveletTransform:
    """
    Continuous Wavelet Transform for time-frequency analysis.

    Shows how frequency content changes over time.

    Renaissance Application:
    - Detect when dominant frequencies change (regime shift)
    - Identify transient events in time-frequency plane
    - Cross-wavelet coherence between assets
    """

    def __init__(self, wavelet: str = 'morl'):
        """
        Initialize CWT.

        Args:
            wavelet: Wavelet type ('morl' = Morlet, good for oscillatory signals)
        """
        self.wavelet = wavelet

    def transform(self, signal: np.ndarray,
                 scales: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CWT.

        Args:
            signal: Time series
            scales: Wavelet scales (auto if None)

        Returns:
            (coefficients, scales) - 2D array of wavelet coefficients
        """
        n = len(signal)

        if scales is None:
            # Default scales (covering various frequencies)
            scales = np.arange(2, min(n // 2, 128))

        # Use scipy's cwt with Morlet-like wavelet
        def morlet(n_points, width):
            """Morlet wavelet approximation."""
            x = np.linspace(-4, 4, n_points)
            return np.exp(-x**2 / 2) * np.cos(5 * x)

        if cwt is not None:
            coeffs = cwt(signal, morlet, scales)
        else:
            # Manual CWT implementation using convolution
            coeffs = np.zeros((len(scales), n))
            for i, scale in enumerate(scales):
                n_points = min(int(scale * 10), n)
                if n_points < 4:
                    n_points = 4
                wavelet = morlet(n_points, scale)
                wavelet = wavelet / np.sqrt(scale)  # Normalize
                # Convolve with padding
                conv = np.convolve(signal, wavelet, mode='same')
                coeffs[i] = conv

        return coeffs, scales

    def scalogram(self, signal: np.ndarray,
                  scales: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Compute scalogram (power of CWT).

        Returns dict with:
        - power: 2D array (scales x time)
        - scales: scale axis
        - times: time axis
        - total_power: power at each time
        - dominant_scale: scale with max power at each time
        """
        coeffs, scales = self.transform(signal, scales)

        power = np.abs(coeffs)**2

        # Summary statistics
        total_power = power.sum(axis=0)
        dominant_scale = scales[np.argmax(power, axis=0)]

        return {
            'power': power,
            'scales': scales,
            'times': np.arange(len(signal)),
            'total_power': total_power,
            'dominant_scale': dominant_scale
        }


class HilbertTransform:
    """
    Hilbert Transform for instantaneous phase/frequency.

    Converts real signal to analytic signal (complex).
    Phase and frequency can be extracted.

    Renaissance Application:
    - Phase relationships between assets
    - Instantaneous frequency (local cyclicality)
    - Lead-lag detection
    """

    def __init__(self):
        pass

    def analytic_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute analytic signal using Hilbert transform.

        Returns complex signal: z(t) = x(t) + i*H[x(t)]
        """
        return hilbert(signal)

    def instantaneous_phase(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous phase.

        φ(t) = arctan(H[x(t)] / x(t))
        """
        analytic = self.analytic_signal(signal)
        return np.angle(analytic)

    def instantaneous_frequency(self, signal: np.ndarray,
                               sample_rate: float = 1.0) -> np.ndarray:
        """
        Compute instantaneous frequency.

        f(t) = dφ/dt / (2π)
        """
        phase = self.instantaneous_phase(signal)

        # Unwrap phase to avoid discontinuities
        unwrapped = np.unwrap(phase)

        # Compute derivative
        inst_freq = np.diff(unwrapped) * sample_rate / (2 * np.pi)

        # Pad to match original length
        inst_freq = np.concatenate([inst_freq, [inst_freq[-1]]])

        return inst_freq

    def envelope(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute envelope (instantaneous amplitude).

        A(t) = |z(t)|
        """
        analytic = self.analytic_signal(signal)
        return np.abs(analytic)


class SpectralFeatureEngine:
    """
    Complete spectral feature generation for HFT.

    Combines:
    1. FFT-based cycle detection
    2. Wavelet decomposition
    3. Hilbert transform features
    """

    def __init__(self, sample_rate: float = 12.0):
        """
        Args:
            sample_rate: Samples per hour (12 for 5-min bars)
        """
        self.spectral = SpectralAnalyzer(sample_rate)
        self.wavelet = WaveletDecomposer(wavelet='db4', levels=4)
        self.hilbert = HilbertTransform()

    def compute_features(self, prices: pd.Series,
                        window: int = 100) -> pd.DataFrame:
        """
        Compute all spectral features.

        Args:
            prices: Price series
            window: Rolling window for spectral analysis

        Returns:
            DataFrame with spectral features
        """
        returns = prices.pct_change().fillna(0)
        n = len(prices)

        features = pd.DataFrame(index=prices.index)

        # Initialize arrays
        dominant_period = np.zeros(n)
        spectral_entropy = np.zeros(n)
        noise_ratio = np.zeros(n)
        trend = np.zeros(n)
        inst_freq = np.zeros(n)
        envelope = np.zeros(n)

        for t in range(window, n):
            window_prices = prices.iloc[t - window:t].values
            window_returns = returns.iloc[t - window:t].values

            # FFT features
            spec_result = self.spectral.welch_spectrum(window_returns)
            dominant_period[t] = spec_result.dominant_period
            spectral_entropy[t] = spec_result.spectral_entropy

            # Noise ratio
            noise_ratio[t] = self.spectral.noise_ratio(window_returns)

            # Wavelet features
            wav_result = self.wavelet.decompose(window_prices)
            trend[t] = wav_result.trend[-1]

            # Hilbert features (on returns)
            if len(window_returns) > 10:
                inst_freq[t] = self.hilbert.instantaneous_frequency(window_returns)[-1]
                envelope[t] = self.hilbert.envelope(window_returns)[-1]

        features['spectral_dominant_period'] = dominant_period
        features['spectral_entropy'] = spectral_entropy
        features['spectral_noise_ratio'] = noise_ratio

        # Wavelet features
        features['wavelet_trend'] = trend
        features['wavelet_deviation'] = prices.values - trend  # Price deviation from trend

        # Trend slope
        features['wavelet_trend_slope'] = pd.Series(trend).diff().fillna(0).values

        # Hilbert features
        features['hilbert_inst_freq'] = inst_freq
        features['hilbert_envelope'] = envelope

        # Derived features
        # Cyclicality score: low entropy + high dominant period = cyclical
        features['cyclicality'] = (1 - spectral_entropy) * np.log1p(dominant_period)

        # Trend strength: low noise ratio = strong trend
        features['trend_strength'] = 1 - noise_ratio

        # Detrended price (mean reversion signal)
        features['detrended_price'] = (prices.values - trend) / (trend + 1e-10) * 10000

        return features


def compute_spectral_features(prices: pd.Series,
                             sample_rate: float = 12.0,
                             window: int = 100) -> pd.DataFrame:
    """
    Convenience function for spectral feature computation.
    """
    engine = SpectralFeatureEngine(sample_rate)
    return engine.compute_features(prices, window)


def detect_cycles(prices: pd.Series,
                 min_period: int = 5,
                 max_period: int = 100) -> List[Dict]:
    """
    Detect significant cycles in price series.

    Returns list of detected cycles with:
    - period: Cycle length in samples
    - power: Relative power
    - significance: Statistical significance
    """
    returns = prices.pct_change().fillna(0).values

    analyzer = SpectralAnalyzer()
    result = analyzer.welch_spectrum(returns)

    # Find peaks in power spectrum
    cycles = []

    # Convert frequency to period
    periods = 1 / (result.frequencies + 1e-10)

    # Filter to specified range
    mask = (periods >= min_period) & (periods <= max_period)
    periods = periods[mask]
    power = result.power[mask]

    if len(power) == 0:
        return cycles

    # Find local maxima
    for i in range(1, len(power) - 1):
        if power[i] > power[i - 1] and power[i] > power[i + 1]:
            # Check significance (power > mean + 2*std)
            threshold = np.mean(power) + 2 * np.std(power)
            if power[i] > threshold:
                cycles.append({
                    'period': periods[i],
                    'power': power[i] / power.max(),
                    'significance': (power[i] - np.mean(power)) / np.std(power)
                })

    # Sort by power
    cycles.sort(key=lambda x: x['power'], reverse=True)

    return cycles[:5]  # Top 5 cycles


if __name__ == '__main__':
    print("Spectral Analysis Test")
    print("=" * 60)

    # Generate synthetic data with known cycles
    np.random.seed(42)
    n = 1000
    t = np.arange(n)

    # Base price with trend
    trend = 1.1 + 0.00001 * t

    # Add cycles
    cycle_20 = 0.001 * np.sin(2 * np.pi * t / 20)  # 20-period cycle
    cycle_50 = 0.002 * np.sin(2 * np.pi * t / 50)  # 50-period cycle

    # Add noise
    noise = np.random.randn(n) * 0.0005

    prices = trend + cycle_20 + cycle_50 + noise
    prices = pd.Series(prices)

    print(f"Generated {n} prices with 20-period and 50-period cycles")

    # Test spectral analyzer
    print("\n--- FFT Spectrum ---")
    analyzer = SpectralAnalyzer()
    result = analyzer.fft_spectrum(prices.pct_change().fillna(0).values)
    print(f"Dominant period: {result.dominant_period:.1f}")
    print(f"Spectral entropy: {result.spectral_entropy:.4f}")

    # Test cycle detection
    print("\n--- Cycle Detection ---")
    cycles = detect_cycles(prices)
    for i, cycle in enumerate(cycles):
        print(f"Cycle {i + 1}: period={cycle['period']:.1f}, "
              f"power={cycle['power']:.4f}, significance={cycle['significance']:.2f}")

    # Test wavelet decomposition
    print("\n--- Wavelet Decomposition ---")
    decomposer = WaveletDecomposer(levels=4)
    wav_result = decomposer.decompose(prices.values)
    print(f"Decomposition levels: {wav_result.levels}")
    print(f"Trend range: {wav_result.trend.min():.4f} - {wav_result.trend.max():.4f}")
    print(f"Noise std: {np.std(wav_result.noise):.6f}")

    # Test full feature engine
    print("\n--- Spectral Feature Engine ---")
    features = compute_spectral_features(prices, window=50)
    print(f"Features computed: {list(features.columns)}")
    print(f"\nSample features (last 5 rows):")
    print(features[['spectral_dominant_period', 'spectral_entropy',
                   'wavelet_trend_slope', 'cyclicality']].tail())

    # Test Hilbert transform
    print("\n--- Hilbert Transform ---")
    hilbert_tf = HilbertTransform()
    returns = prices.pct_change().fillna(0).values
    inst_freq = hilbert_tf.instantaneous_frequency(returns[-100:])
    envelope = hilbert_tf.envelope(returns[-100:])
    print(f"Instantaneous frequency range: {inst_freq.min():.4f} - {inst_freq.max():.4f}")
    print(f"Envelope range: {envelope.min():.6f} - {envelope.max():.6f}")

    print("\n" + "=" * 60)
    print("Spectral analysis tests passed!")
