# PLAN.md — Rencana Kerja `quantum-chaos-framework`

> **Status:** DISETUJUI 2026-10-14 ("setuju semua rekomendasi"). Keputusan dicatat di §10.
> **Basis audit:** commit `69396f5` (branch `main`, working tree bersih).
> **Tanggal:** 2026-10-14

Dokumen ini berisi (1) hasil audit repo, termasuk bug yang sudah saya **reproduksi secara numerik**;
(2) rencana per fase beserta test yang harus gagal di kode lama; (3) target validasi beserta toleransinya;
(4) keputusan yang perlu Anda ambil sebelum saya mulai (§10).

Konvensi penandaan:
- **[TERVERIFIKASI]**: sudah saya jalankan dan hasilnya dicantumkan.
- **[DIBACA]**: disimpulkan dari membaca kode, belum dijalankan.
- **PERLU VERIFIKASI**: nilai atau klaim yang belum saya cek terhadap sumber primer.

---

## 1. Ringkasan audit

### 1.1 Cakupan
Saya membaca seluruh 24 file yang di-track git (≈ 3.960 baris). Verifikasi numerik dijalankan
dengan skrip di direktori scratch di luar repo:
- Python 3.14.6 (Anaconda), NumPy 2.4.6, SciPy 1.18.0, untuk jalur NumPy;
- venv terpisah berisi **PennyLane 0.45.1** (versi terbaru di PyPI), NumPy 2.5.4, untuk jalur PennyLane;
- Python 3.12.6, untuk menguji perilaku anotasi saat import (lihat §1.2 butir 1).

Mesin lokal: Apple M1, 8 core, RAM 16 GB. Ini membatasi target skala di Fase 3A (lihat §7A).

### 1.2 Bug dari brief: status konfirmasi

| # | Bug | Status | Bukti |
|---|-----|--------|-------|
| 1 | API PennyLane usang: `qml.Device` (anotasi return di `TrotterEvolution.device`, `NISQSimulator.ideal_device/noisy_device`, `QuantumKernel.device`), `qml.QubitStateVector`, `qml.operation.Tensor` | **[TERVERIFIKASI]**, dengan koreksi pada angka | Di PennyLane 0.45.1, `qml.Device`, `qml.QubitStateVector`, dan `qml.operation.Tensor` **tidak ada lagi**. **Saat import:** di Python ≤3.13, anotasi `-> qml.Device` langsung dievaluasi dan memunculkan `AttributeError` (mekanisme ini saya reproduksi di 3.12.6). Subpaket yang gagal adalah **3 dari 6**: `circuits`, `noise`, dan `qml`. Melalui rantai `__init__`, ini berarti 6 dari 11 file modul. `hamiltonians`, `utils`, dan `visualization` tidak bergantung pada PennyLane. Di Python 3.14, anotasi dievaluasi malas (PEP 649), sehingga **semua subpaket ter-import**, tetapi gagal saat runtime. **Saat runtime (3.14 + PL 0.45.1):** `hamiltonian_to_pennylane` gagal karena `Tensor` tidak ada; `evolve_state(initial_state=…)` dan `create_pennylane_noisy_circuit` gagal karena `QubitStateVector` tidak ada. Selain itu, `shots=` pada `qml.device` memunculkan `PennyLaneDeprecationWarning` (yang baru adalah transform `set_shots` pada QNode). Saya tidak menemukan konfigurasi yang menghasilkan "5 dari 6". Angka yang benar akan dicatat di PR. |
| 2 | `partial_trace` salah | **[TERVERIFIKASI]** | `Tr_B(ρ_A⊗ρ_B)` ≠ `ρ_A`, dan trace hasilnya 0,263 (seharusnya 1). Untuk state produk murni `\|0⟩\|1⟩`, hasilnya **matriks nol**. **Catatan penting:** `entanglement_entropy` untuk state produk tetap memberi 0, dan untuk state Bell tetap memberi 1, **karena kebetulan** (eigenvalue matriks nol dibuang oleh filter `> 1e-12`). Jadi test entropi saja **tidak gagal** di kode lama. Test yang gagal-dulu adalah `Tr_B(ρ_A⊗ρ_B)=ρ_A` dan `Tr=1`. |
| 3 | Fidelity di `compare_ideal_vs_noisy` memakai `np.sqrt` per elemen | **[TERVERIFIKASI]** | Rumus lama memberi F(\|+⟩,\|+⟩) = **4,0000** dan F(\|+i⟩,\|+i⟩) = 3,4142. Hasilnya hanya benar untuk state diagonal. |
| 4 | Tanda quasi-energi terbalik | **[TERVERIFIKASI]** | Dengan `J₀=0`, `U=1,3`, `ω=10`: kode memberi {−2,6; −1,3; 0}, sedangkan yang benar (eigenvalue H setelah dilipat) adalah {0; 1,3; 2,6}. |
| 5 | `apply_channel` crash untuk ρ real | **[TERVERIFIKASI]** | `UFuncTypeError: Cannot cast ufunc 'add' output from complex128 to float64` |
| 6 | Phase damping di `full_noise_model` | **[TERVERIFIKASI]** | Dengan T1=100, T2=80, t=20: \|ρ01\| = 0,39926, padahal exp(−t/T2)/2 = 0,38940. Laju efektif yang terjadi adalah 1/(2T1)+1/(2T2), bukan 1/T2. Populasi (T1) sudah benar. |
| 7 | `HubbardKernel` selalu bernilai 1 | **[TERVERIFIKASI]** | Matriks kernel untuk 6 titik acak semuanya bernilai 1. Penyebabnya: \|0000⟩ adalah vakum, yaitu eigenstate dari hopping maupun interaksi. |
| 8 | `SYKKernel` hanya bergantung pada `mean(x)` | **[TERVERIFIKASI]** | K(x₁,x₂) = 1,0000 untuk x₁=(1,−1,2,0) dan x₂=(½,½,½,½), dua vektor berbeda dengan rata-rata sama. |
| 9 | `create_pennylane_kernel`: K(x,x)≠1 | **[TERVERIFIKASI]** | Untuk 3 qubit dan x acak, K(x,x) = 0,136, 0,111, dan 0,261. Sirkuitnya RY·RZ(x₁) → tangga CNOT → RY·RZ(−x₂), yang bukan U(x₂)†U(x₁). |
| 10 | Placeholder yang menyamar sebagai fitur | **[TERVERIFIKASI]** (sebagian [DIBACA]) | `create_otoc_circuit` memberi 1,0000 baik di t=0 maupun t=5 (parameter waktu tidak dipakai sama sekali). `effective_hamiltonian(order)` mengabaikan `order` (order=1 dan order=5 identik). `error_mitigation_zne` selalu memakai `result = 0.0`, dan `create_noisy_circuit` hanya menambah noise di akhir [DIBACA]. Detail per fungsi ada di §4, P0-9. |
| 11 | Hal minor: shebang `#!/usr/bin/env pyt`, URL clone `yourusername`, 3× bare `except:` (`otoc_calculator.py:205`, `syk_hamiltonian.py:248`, `quantum_visualizer.py:26`), RNG global, komentar "Unfold" | [DIBACA] | `np.random.seed` atau RNG global dipakai di `helpers.py:14`, `quantum_classifier.py:127`, `trotter_evolution.py:269`, `nisq_simulator.py:109,161`, `quantum_kernel.py:84`, dan `noise_channels.py:158`. |

### 1.3 Temuan tambahan (di luar brief)

| # | Temuan | Status | Dampak |
|---|--------|--------|--------|
| A1 | **Kelas simetri SYK salah dibandingkan.** `main.py` dan gambar `output.png` membandingkan N=6 dan N=10 dengan GOE, padahal N≡6 dan N≡2 (mod 8) termasuk GUE. | **[TERVERIFIKASI]** | Lihat tabel di §1.4. |
| A2 | Filter `spacing > 1e-12` **kebetulan** memberi hasil GSE yang benar untuk N=12 (sektor genap: 51,6% spasi bernilai nol akibat pasangan Kramers). Tanpa filter, ⟨r⟩ menjadi 0. Kebenaran ini tidak disengaja dan akan rusak untuk degenerasi aksidental. | **[TERVERIFIKASI]** | Fase 1 |
| A3 | `floquet_operator` memakai produk titik-kiri (orde-1). Dengan 100 langkah (L=2, ω=2), ‖F₁₀₀ − F_ref‖₂ = **1,6·10⁻²**. | **[TERVERIFIKASI]** | Fase 0 (P0-4) |
| A4 | Konvensi **depolarizing tidak konsisten**. Jalur NumPy memakai ρ→(1−p)ρ+p·I/2 (**[TERVERIFIKASI]**: p=1 memberi I/2), sedangkan `qml.DepolarizingChannel(p)` di PennyLane memakai ρ→(1−p)ρ+(p/3)(XρX+YρY+ZρZ). Parameter `depolarizing_rate` yang sama berarti dua channel berbeda. `estimate_circuit_depth_limit` dan `summary()` menyebut 1−3p/4 sebagai "gate fidelity", padahal itu fidelity keterikatan. Fidelity rata-rata untuk konvensi NumPy adalah 1−p/2. | [DIBACA]/sebagian [TERVERIFIKASI] | Fase 0 (P0-6) |
| A5 | **Satuan waktu noise tidak konsisten.** `compare_ideal_vs_noisy` memakai `n_steps=int(10t)`, dan setiap langkah menerapkan T1/T2 dengan `gate_time` tetap. Akibatnya dekoherensi total ∝ jumlah langkah, bukan ∝ waktu fisik t. | [DIBACA] | Fase 0 (P0-6). Test: untuk H=0, peluruhan tidak boleh bergantung pada `n_steps`. |
| A6 | `plot_trotter_error`: label "O(t²/n)" dan garis referensi 1/n, 1/n² salah untuk **infidelity**. Infidelity berskala sebagai kuadrat galat state, yaitu ∝ n⁻² (orde-1) dan n⁻⁴ (orde-2). | [DIBACA] | P0-10 |
| A7 | **Metrik yang menyesatkan:** `benchmark_spectral_stability` (`ground_state_error = √var`, dengan komentar "variational principle" yang keliru), `compute_noisy_eigenspectrum` (bukan spektrum, hanya energi state acak setelah satu lapis noise), `expressivity_score = 1−mean(K)` (kontradiktif dengan konsentrasi kernel), `chaos_advantage`, `_draw_conclusion` ("significant advantage" dengan ambang 0,05 tanpa uji statistik), dan teks hard-coded "Chaos Advantage: Higher expressivity for QML" di `create_summary_dashboard`. | [DIBACA] | Melanggar aturan 4. Akan dihapus di P0-9. |
| A8 | `lyapunov_exponent` mem-fit log(1−Re C) pada jendela berupa fraksi indeks waktu, untuk N≤8 dan T=∞. Angka yang dihasilkan tidak punya makna kontrol. `butterfly_velocity` tidak terdefinisi untuk SYK karena modelnya all-to-all (tidak ada jarak). | [DIBACA] | P0-9, lalu ditulis ulang di Fase 3B |
| A9 | Konvensi Majorana: Jordan–Wigner di sini memberi χ²=1 ({χᵢ,χⱼ}=2δᵢⱼ), sedangkan varians kopling ⟨J²⟩=3!J²/N³ mengikuti Maldacena–Stanford (yang memakai χ²=½). Akibatnya H_kode = 4·H_MS. Tr H²/D konsisten dengan C(N,4)·6/N³ (normalisasi χ²=1) **[TERVERIFIKASI, per realisasi]**. Faktor ini **wajib** didokumentasikan sebelum membandingkan E₀ dengan literatur. | sebagian [TERVERIFIKASI] | Fase 1 dan 3B |
| A10 | `use_sparse=True` di SYK tetap membangun matriks **dense** dulu, lalu dikonversi. Tidak ada penghematan memori. | [DIBACA] | Fase 3A |
| A11 | `.gitignore` mengabaikan `*.ipynb`, sehingga bertentangan dengan tutorial notebook di Fase 2. README memakai GUE ≈ 0,603 (nilai surmise), bukan 0,5996 (Atas et al.). README juga mengklaim SYK "saturates chaos bound" tanpa syarat (seharusnya hanya pada limit N besar dan kopling kuat). | [DIBACA] | Fase 0 dan 1 |
| A12 | `trotter_evolution._pauli_rotation` **benar**. Basis Y→Z via RX(±π/2) benar secara analitik. Secara numerik, galat state vs `expm` berskala n⁻¹,⁰ (orde-1) dan n⁻²,⁰ (orde-2) untuk H 2-qubit dengan suku XY/YX. | **[TERVERIFIKASI]** | Jadikan test regresi, tidak perlu perbaikan |
| A13 | `get_pauli_string_representation` berkompleksitas O(16ⁿ·…) karena membangun setiap string Pauli secara dense. | [DIBACA] | Fase 3 (efisiensi) |
| A14 | Bobot Boltzmann di OTOC dan `get_thermal_state` tidak di-shift dengan E₀, sehingga bisa overflow untuk β\|E\| besar. | [DIBACA] | Fase 3B (perlu juga ditambah ke P0-10) |

### 1.4 Statistik spektral SYK dari kode **lama** (baseline) [TERVERIFIKASI]

Sektor paritas genap; 20 realisasi untuk N≤14 dan 5 realisasi untuk N=16; tanpa pemangkasan tepi spektrum.

| N | N mod 8 | Kelas yang diharapkan | dim sektor | ⟨r⟩ kode lama (dengan filter) | ⟨r⟩ tanpa filter | fraksi spasi degenerat |
|---|---|---|---|---|---|---|
| 6 | 6 | GUE | 4 | 0,561 ± 0,038 | 0,561 | 0 |
| 8 | 0 | GOE | 8 | 0,509 ± 0,037 | 0,509 | 0 |
| 10 | 2 | GUE | 16 | 0,580 ± 0,012 | 0,580 | 0 |
| 12 | 4 | GSE | 32 | 0,667 ± 0,012 | **0,000** | **0,516** |
| 14 | 6 | GUE | 64 | 0,599 ± 0,007 | 0,599 | 0 |
| 16 | 0 | GOE | 128 | 0,527 ± 0,009 | 0,527 | 0 |

Kesimpulan: fisika Hamiltonian SYK tampak benar. Yang salah adalah **klasifikasi dan pembandingnya**, ditambah N yang terlalu kecil (N=6 hanya memberi 2 rasio per realisasi, sehingga tidak bermakna statistik). Hubbard L=2 memberi ⟨r⟩=0,7808 dari 16 level campuran semua sektor **[TERVERIFIKASI]**. Angka ini tidak bermakna.

---

## 2. Prinsip kerja dan Definition of Done per PR

1. **Test gagal-dulu.** Untuk setiap bug, commit pertama menambah test dengan
   `@pytest.mark.xfail(strict=True, reason="bug: …")`. Suite tetap hijau, dan kegagalan di kode lama
   terdokumentasi. Commit kedua memperbaiki bug dan menghapus `xfail`. Karena `strict=True`, test
   akan gagal jika lulus tak terduga, sehingga penanda tidak bisa lupa dihapus. Output kegagalan
   di kode lama disalin ke deskripsi PR.
2. **Satu fase = satu branch = satu PR** (`fix/phase-0-…`, `fix/phase-1-spectral`, `chore/phase-2-packaging`,
   `feat/phase-3x-…`). Commit kecil dan atomik. Fase 3 dipecah per sub-bagian (A–E).
3. **Tanpa angka karangan.** Setiap konstanta literatur di kode punya docstring berisi sitasi
   lengkap. Nilai yang belum dicek ditulis `PERLU VERIFIKASI` dan **tidak** dipakai sebagai ambang test.
4. **Toleransi dinyatakan eksplisit** di setiap test numerik: atol/rtol untuk identitas eksak, dan
   kσ statistik ditambah offset ukuran-hingga untuk besaran stokastik (§8).
5. **RNG eksplisit:** semua fungsi stokastik menerima `rng: np.random.Generator | int | None`.
   Tidak ada lagi state global.
6. **Hasil negatif dilaporkan.** Kata "advantage", "significant", dan "maximally chaotic" tidak boleh
   muncul di kode, README, atau gambar tanpa uji statistik dan baseline.

---

## 3. Fase "pra-0": infrastruktur test (kecil, perlu persetujuan, lihat Q1)

Supaya test bisa ditulis sekali dan tidak perlu ditulis ulang di Fase 2, rekomendasi saya adalah
memindahkan kode ke `src/quantum_chaos/` **secara mekanis** sebelum Fase 0: `git mv`, memperbaiki
import, dan menghapus hack `try: from .. / except ImportError: from utils`. Isi pra-0:
`pyproject.toml` minimal (`pip install -e .`), `pytest` dan konfigurasi dasarnya, tanpa perubahan
perilaku. Ruff, mypy, CI, dan dokumentasi tetap di Fase 2.

Alternatif: jika Anda ingin struktur tetap datar sampai Fase 2, saya pakai `tests/conftest.py`
yang menambahkan root repo ke `sys.path`, lalu import di test diperbarui saat Fase 2.

---

## 4. Fase 0: perbaikan bug terkonfirmasi

Urutan commit di dalam PR Fase 0. Setiap butir menyebut test yang gagal-dulu (FD) dan test
regresi (R).

**P0-1 · Kompatibilitas PennyLane**
- Hapus anotasi `qml.Device` (pakai `qml.devices.Device` jika anotasi masih diperlukan). Ganti
  `QubitStateVector` → `qml.StatePrep` dan `Tensor` → `qml.prod`. Pindahkan `shots` dari
  `qml.device(...)` ke `qml.set_shots` (deprecated di 0.45.1, [TERVERIFIKASI]). Pin
  `pennylane>=0.45,<0.46` sesuai versi yang diuji (rentang final mengikuti jawaban Q2).
- FD: `test_imports.py` meng-import setiap subpaket. Test ini gagal di Python ≤3.13; di 3.14 lulus
  karena PEP 649, sehingga di 3.14 kegagalan ditangkap oleh dua FD berikut.
- FD: `evolve_state(initial_state=…)` dan `hamiltonian_to_pennylane` dengan suku multi-qubit
  (keduanya gagal di semua versi Python dengan PL 0.45.1). Pengujian dilakukan dengan
  `filterwarnings = error::PennyLaneDeprecationWarning`.
- R: `TrotterEvolution` orde-1/2 untuk Hamiltonian Pauli acak (termasuk suku Y) dibandingkan
  dengan `expm`. Galat harus berskala n⁻¹ / n⁻² (slope log-log dalam ±0,15).
  `hamiltonian_to_pennylane(terms).matrix()` harus sama dengan matriks input (atol 1e-12).

**P0-2 · `partial_trace` dan `entanglement_entropy`**
- Tulis ulang dengan reshape + `np.einsum`/`transpose`. Validasi bahwa `dims` dan bentuk ρ cocok.
- FD: `Tr_B(ρ_A⊗ρ_B)=ρ_A` untuk ρ acak (2×2, 2×3, dan 3 subsistem dengan `keep` tak berurutan);
  `Tr(hasil)=1`; ρ_A dari state produk murni sama dengan proyektor yang benar (bukan matriks nol).
- R: S(produk)=0, S(Bell)=1 bit, S(GHZ₃, 1 qubit)=1. Untuk state murni acak, S_A=S_B.
  (Test S(produk) dan S(Bell) sudah lulus di kode lama karena kebetulan, jadi statusnya regresi, bukan FD.)

**P0-3 · Fidelity**
- Satu implementasi `fidelity(ρ,σ)` di `utils`, memakai akar matriks berbasis `eigh` dengan
  eigenvalue negatif-kecil dipotong. Pertimbangannya: `scipy.linalg.sqrtm` tidak stabil untuk ρ
  berpangkat rendah. Brief meminta `sqrtm`; jika Anda ingin `sqrtm` secara harfiah, keduanya akan
  diuji setara. Dipakai oleh `NISQSimulator`.
- FD: F(ρ,ρ)=1 untuk ρ murni non-diagonal (kode lama memberi 4).
- R: F(\|ψ⟩,\|φ⟩)=\|⟨ψ\|φ⟩\|²; F simetris; 0≤F≤1; untuk ρ,σ diagonal, F = (Σ√(pᵢqᵢ))².

**P0-4 · Floquet**
- Gunakan ε = −angle(λ)/T yang dilipat ke [−ω/2, ω/2). Propagator diganti ke aturan titik-tengah
  (orde-2) dengan uji konvergensi, karena orde-1 memberi galat 1,6e-2 (A3).
- FD: untuk H konstan (J₀=0), quasi-energi = eigenvalue H yang dilipat (atol 1e-10).
- R: F(J₀=0) = expm(−iH_int·T) eksak; ‖F_n − F_ref‖ ∝ n⁻² (slope dalam ±0,15); F uniter (atol 1e-12).

**P0-5 · `apply_channel` dtype**
- Paksa `complex128`.
- FD: ρ real tidak menyebabkan crash dan hasilnya sama dengan ρ yang di-cast ke kompleks.
- R: setiap Kraus set memenuhi Σ K†K = I; channel mempertahankan trace dan hermitisitas.

**P0-6 · Model noise T1/T2 dan konsistensi**
- 1/T_φ = 1/T2 − 1/(2T1), γ_φ = 1 − exp(−2·Δt/T_φ). Faktornya 2 karena Kraus phase-damping
  mengalikan ρ01 dengan √(1−γ). Validasi T2 ≤ 2T1 (`ValueError` jika dilanggar).
  Satuan waktu dibuat eksplisit: noise per langkah memakai Δt fisik (A5).
  Konvensi depolarizing disatukan dan didokumentasikan (A4); nama "gate fidelity" dikoreksi.
- FD: untuk H=0, \|ρ01(t)\| = ½·exp(−t/T2) (rtol 1e-10). Kode lama memberi 0,39926 vs 0,38940.
- FD: hasil tidak bergantung pada jumlah langkah (n langkah Δt sama dengan 1 langkah n·Δt) (A5).
- R: ρ11(t) = ρ11(0)·exp(−t/T1); jalur NumPy dan `default.mixed` PennyLane memberi ρ yang sama
  untuk sirkuit acak kecil (atol 1e-10).

**P0-7 · Kernel kuantum** (desain encoding perlu persetujuan, lihat Q3)
- Usulan *Hamiltonian encoding* yang seragam untuk kedua model: H(x) = H_model + Σⱼ xⱼ Oⱼ, dengan
  Oⱼ densitas lokal (Zⱼ = iχ₂ⱼχ₂ⱼ₊₁ untuk SYK; nⱼ↑+nⱼ↓ untuk Hubbard). Evolusi selama waktu t dari
  state referensi tetap. Untuk Hubbard, state referensi adalah Fock half-filling (Néel), sehingga
  setiap fitur terpakai dan kedua model dibandingkan secara apel-dengan-apel. Jika jumlah fitur >
  jumlah situs, dipakai beberapa lapis (re-uploading).
  Alasan menolak rotasi Z lokal biasa: pada state Fock, rotasi itu hanya menambah fase global,
  sehingga kernel kembali trivial.
- `create_pennylane_kernel` = probabilitas \|0…0⟩ setelah U(x₁) lalu `qml.adjoint`(U(x₂)), dengan
  feature map yang **sama** seperti jalur NumPy.
- FD: K(x,x)=1 (sirkuit lama ≠ 1); Hubbard: Var[K] > 0 pada data acak (kode lama: semuanya 1);
  SYK: K(x₁,x₂) < 1 − 10⁻³ untuk x₁≠x₂ dengan mean sama (kode lama: 1).
- R: K simetris dan PSD (eigenvalue minimum ≥ −1e-10); kernel PennyLane = kernel NumPy (atol 1e-8).

**P0-8 · `HubbardKernel` state awal** sudah tercakup di P0-7.

**P0-9 · Fungsi palsu dan menyesatkan.** Rekomendasi per fungsi (lihat Q4):

| Fungsi | Tindakan | Test |
|---|---|---|
| `create_otoc_circuit` | **Implementasi minimal yang benar.** Uji Hadamard memakai identitas c-(U†XU) = U†·cX·U, sehingga evolusi waktu **tidak** perlu dikontrol (Swingle et al. 2016). U(t) = `qml.QubitUnitary` eksak. T=∞ via purifikasi (n pasangan Bell). Versi Trotter menyusul di Fase 3D. | Re C(t) dari sirkuit = `compute_otoc_exact` (atol 1e-8) untuk W,V Pauli acak, baik state murni maupun T=∞ |
| `error_mitigation_zne` (mengembalikan 0) | **Hapus.** ZNE nyata di Fase 3D | Test API memastikan fungsi tidak ada |
| `create_noisy_circuit` (noise hanya di akhir) | **Hapus** (diganti oleh fungsi berikut) | — |
| `create_pennylane_noisy_circuit` (hanya suku `ZZZZ`) | **Perbaiki:** pakai lapisan Trotter penuh dari `TrotterEvolution` + noise per lapis | Noise=0 sama dengan state Trotter ideal; dengan noise, cocok dengan jalur NumPy |
| `effective_hamiltonian(order)` | Pertahankan **hanya** orde terendah (rata-rata waktu). Ini **memang benar** untuk drive J₀cos ωt yang rata-ratanya nol. Orde lain memunculkan `NotImplementedError` sampai Fase 3C | Orde 0 = (1/T)∫H dt (numerik); orde lain memunculkan error |
| `lyapunov_exponent`, `compare_models`, `butterfly_velocity`, `scrambling_time` | Hapus dari API publik (A8). Ditulis ulang dengan estimasi galat di Fase 3B | — |
| `benchmark_spectral_stability`, `compute_noisy_eigenspectrum` | Hapus atau ganti nama secara jujur (A7) | — |
| `expressivity_score`, `chaos_advantage`, `_draw_conclusion`, teks "Chaos Advantage" di dashboard, label "(Regular)" atau "(Chaotic)" yang hard-coded | Hapus. Label jadi berbasis data | — |

**P0-10 · Minor**
- Shebang; URL clone → `rasidi3112`; ganti bare `except:` dengan exception spesifik; RNG → `default_rng`.
  Catatan: ini **mengubah** realisasi acak untuk seed yang sama, dan akan dicatat di CHANGELOG.
- Komentar "Unfold" diganti nama. Unfolding yang sebenarnya dikerjakan di Fase 1.
- Slope referensi di plot Trotter (A6); `print` peringatan → `warnings.warn`; `level_spacing_ratio`
  mengembalikan `nan` atau memunculkan error untuk input < 3 level, bukan `0.0`; shift E₀ di bobot
  Boltzmann (A14).
- README: hapus sementara `output.png` beserta klaim GOE untuk N=6,10. Gambar diregenerasi di akhir Fase 1.

**Kriteria selesai Fase 0:** semua test lulus; tidak ada `xfail` tersisa; setiap modul bisa di-import
dengan PennyLane versi yang di-pin; tidak ada fungsi placeholder (dicek dengan grep `placeholder|simplified|TODO`).

---

## 5. Fase 1: koreksi fisika statistik spektral

**P1-1 · Modul referensi RMT** (`spectral/rmt.py`)
- Konstanta dari Atas et al., PRL 110, 084101 (2013): ⟨r⟩ Poisson = 2ln2−1 ≈ 0,3863 (eksak),
  GOE 0,5307, GUE 0,5996, GSE 0,6744. Nilai-nilai ini berasal dari brief Anda dan cocok dengan
  ingatan saya; tetap akan dicek ke paper sebelum merge.
- Sampler ansambel berukuran hingga (GOE/GUE/GSE; GSE via matriks kuaternion dengan deduplikasi
  Kramers) memberi **referensi yang dicocokkan dengan dimensi dan pemangkasan yang sama** dengan
  data SYK atau Hubbard. Ini memisahkan efek ukuran-hingga dari deviasi fisik.
- Test: sampel GOE/GUE/GSE (dim ≈ 400, ≥ 200 sampel) mereproduksi nilai Atas dalam 3σ
  ditambah offset ukuran-hingga yang terukur; sampel Poisson mereproduksi 0,3863.

**P1-2 · `level_spacing_ratio` baru**
- Tanpa filter `1e-12`. Argumen eksplisit `kramers=True/False` (ambil satu dari setiap pasangan
  **setelah** memeriksa bahwa pasangan itu memang degenerat dalam toleransi). Pemangkasan tepi
  spektrum (misalnya hanya fraksi tengah) bisa dikonfigurasi. Mengembalikan array rasio per level
  untuk analisis galat.
- Degenerasi yang tidak dijelaskan simetri memunculkan peringatan atau error, bukan dibuang diam-diam.
- Test: input dengan pasangan degenerat dan `kramers=True` sama dengan spektrum unik; degenerasi
  eksak tanpa deklarasi memicu peringatan.

**P1-3 · Kelas simetri SYK berdasarkan N mod 8**
- `syk_symmetry_class(N)`: 0 → GOE, 2/6 → GUE, 4 → GSE (per sektor paritas). Referensi RMT dipilih otomatis.
- Konstruksi eksplisit operator partikel-lubang antiuniter P, lalu uji struktural: [P,H]=0, P²=±1
  sesuai N mod 8, dan P mempertahankan atau menukar sektor paritas.
- Test spektral: N≡2,6 → spektrum kedua sektor identik (atol 1e-10); N≡4 → pasangan Kramers di
  setiap sektor; N≡0 → tidak ada degenerasi sistematis.
- Rata-rata disorder dengan galat bootstrap. Untuk N≡0, rasio dari kedua sektor dihitung
  **terpisah** lalu dirata-ratakan (spektrum dua sektor tidak pernah digabung).
- Test statistik (N=12 GSE, N=14 GUE, N=16 GOE, dan N=18/20 jika waktu CI memungkinkan, dengan
  label `slow`): \|⟨r⟩ − ref_dim-matched\| < 3σ (§8).

**P1-4 · Hubbard ED per sektor simetri** (modul baru, menggantikan konstruksi dense 4^L)
- Basis bitstring per (N↑, N↓), H sparse, translasi dengan tanda fermionik (PBC) → momentum k;
  refleksi di k=0,π (atau OBC); spin total S via diagonalisasi S² di dalam blok; η-spin untuk t'=0.
  Hopping NNN t' memecah integrabilitas dan juga memecah η-SU(2) karena menghubungkan sublattice
  yang sama.
- Catatan fisika: di sektor k≠0,π, blok H kompleks, tetapi ada simetri antiuniter TR∘refleksi
  dengan kuadrat +1. Jadi kelas yang diharapkan untuk t'≠0 adalah **GOE**, bukan GUE.
- Ukuran: L=8 (half filling: 4.900 state, ≈600 per k) dan L=10 (63.504, ≈6.350 per k; diagonalisasi
  dense per blok ≲ 650 MB, layak di 16 GB). L=6 hanya untuk test cepat. Untuk PBC, t' butuh L≥5
  supaya tidak terjadi double counting.
- Test eksak: U=0 → energi banyak-partikel = jumlah ε_k = −2t cos k (ditambah suku t'); t=0 →
  energi = U·(jumlah doublon); untuk L≤5, gabungan spektrum semua sektor = spektrum ED penuh;
  [H, operator simetri] = 0; untuk L=2,3, spektrum cocok dengan implementasi JW lama (validasi silang).
- Kontrol: t'=0 (integrabel, diharapkan mendekati Poisson setelah resolusi lengkap) vs t'/t ∈ {0,2; 0,5}
  (diharapkan GOE). **PERLU VERIFIKASI:** seberapa dekat ⟨r⟩ ke Poisson pada L≤10 untuk Hubbard 1D
  integrabel. Degenerasi sisa akibat integrabilitas mungkin muncul, dan akan dilaporkan apa adanya,
  bukan dipaksakan.

**P1-5 · Integrasi:** `main.py` dan README memakai referensi yang benar per kelas. `output.png`
diregenerasi. Draf awal `VALIDATION.md` (tabel ⟨r⟩ SYK per N dan Hubbard t'=0 vs t'≠0).

**Kriteria selesai Fase 1:** ⟨r⟩ SYK untuk ketiga kelas (GOE, GUE, GSE) direproduksi dalam toleransi §8;
Hubbard menunjukkan perbedaan yang terukur antara t'=0 dan t'≠0, dengan galat; tidak ada filter spasi.

---

## 6. Fase 2: rekayasa perangkat lunak kelas riset

- Struktur `src/quantum_chaos/{hamiltonians,spectral,dynamics,noise,qml,viz,utils}` (jika pra-0
  disetujui, sebagian sudah selesai). `pyproject.toml` dengan batas versi:
  `numpy>=1.26,<3`, `scipy>=1.11,<2`, `pennylane` di-pin ke rentang yang diuji, dan
  `matplotlib`, `scikit-learn`. Extras: `[qiskit]`, `[mitiq]`, `[docs]`, `[dev]`.
- **Python 3.10–3.13: ada konflik [TERVERIFIKASI via metadata PyPI].** PennyLane 0.42 adalah versi
  terakhir yang mendukung Python 3.10 (`>=3.10`). Versi 0.43, 0.44, 0.45.x (terbaru, 0.45.1) mensyaratkan
  `>=3.11`. Menurut PEP 619, Python 3.10 juga mencapai EOL pada Oktober 2026. Pilihannya ada di Q2.
- pytest + pytest-cov (≥ 85%, gate di CI); test lambat ditandai `@pytest.mark.slow` dan dijalankan
  di job terpisah atau terjadwal. `hypothesis` untuk property test (partial trace, channel CPTP).
- ruff (lint + format), mypy (strict untuk `spectral`, `utils`, `noise`), pre-commit.
- GitHub Actions: matriks Python × {ubuntu, macos}, ditambah job mingguan melawan PennyLane terbaru
  untuk deteksi dini perubahan API.
- Dokumentasi MkDocs-Material + mkdocstrings (API reference). Tutorial notebook memakai format
  jupytext `.py:percent` (diff bersih) dan dieksekusi di CI via mkdocs-jupyter. `.gitignore`
  untuk `*.ipynb` disesuaikan.
- `CITATION.cff` (perlu nama, ORCID, dan afiliasi Anda, lihat Q7), CHANGELOG (Keep a Changelog), SemVer
  (Q9). Rilis Zenodo dan DOI **membutuhkan tindakan Anda** (menghubungkan repo GitHub ke Zenodo).
- Konfigurasi eksperimen YAML dengan validasi dataclass, atau Hydra jika Anda memilihnya (Q8).
  Seed eksplisit di setiap config. `make figures` meregenerasi **setiap** gambar dan menulis
  metadata (git hash, hash config, versi paket) ke sidecar JSON.

---

## 7. Fase 3: skala dan fitur riset (setelah Fase 0–2 lulus)

**A. Skala numerik.** Estimasi berikut perlu dikonfirmasi lewat benchmark.
- Full ED per sektor paritas: N=28 (dim 8.192) mudah; N=30 (16.384, matriks dense kompleks ≈ 4,3 GB)
  layak di 16 GB; **N=32 dense (≈ 17 GB) tidak layak di mesin ini.**
- H SYK di basis JW **tidak terlalu sparse**: perkiraan nnz per baris ≈ C(N/2,4) + C(N/2,2) + 1.
  Untuk N=34 itu ≈ 2.517, sehingga CSR per sektor ≈ 1,6·10⁸ nnz ≈ 3 GB. Rencana: matvec
  *matrix-free* yang dikelompokkan per mask bit-flip, Lanczos untuk tepi spektrum, dan dynamical
  typicality (Bartsch & Gemmer 2009) untuk dinamika di N≈32–34.
- Shift-invert di tengah spektrum kemungkinan besar terhalang oleh fill-in faktorisasi LU untuk
  N≳28. Alternatif yang akan dibenchmark adalah POLFED (Sierant, Lewenstein, Zakrzewski 2020)
  atau filter Chebyshev. Hasil benchmark (waktu dan memori vs N) didokumentasikan, termasuk N
  maksimum yang benar-benar tercapai.

**B. Diagnostik chaos**
- SFF dengan rata-rata disorder (dip, ramp, plateau). Thouless time didefinisikan secara eksplisit.
  Bentuk ramp dibandingkan dengan form factor RMT dari sampel ansambel di P1-1, dan secara
  kualitatif dengan Cotler et al. (JHEP 2017).
- OTOC suhu hingga, versi biasa maupun teregularisasi (ρ^{1/4} sudah benar secara struktur).
  Ekstraksi λ_L dengan galat dari variasi jendela fit dan bootstrap disorder, lalu dibandingkan
  dengan batas MSS λ ≤ 2πT. **Ekspektasi jujur:** pada N≤34, jendela eksponensial pendek, sehingga
  saturasi batas MSS **tidak** bisa diklaim. Pembanding untuk N besar: Kobrin et al. (PRL 2021)
  (PERLU VERIFIKASI detail).
- Koefisien Lanczos dan kompleksitas Krylov (Parker et al., PRX 2019); distribusi ukuran operator
  (Roberts, Stanford, Streicher, JHEP 2018, PERLU VERIFIKASI).
- ETH: elemen matriks diagonal dan off-diagonal serta skalanya terhadap D. Entropi keterikatan
  eigenstate dibandingkan dengan kurva Page (Page 1993), dengan catatan bahwa kendala paritas
  fermion bisa menggeser nilai rata-rata (PERLU VERIFIKASI literatur spesifik SYK).
- Validasi DOS dan E₀ per N terhadap García-García & Verbaarschot (PRD 2016) dan Maldacena & Stanford
  (PRD 2016), setelah konversi konvensi χ²=1 ↔ χ²=½ (faktor 4 pada energi, A9). Nilai referensi akan
  diambil langsung dari paper. **Tidak ada nilai E₀ yang dicantumkan di rencana ini** karena belum saya verifikasi.

**C. Floquet Hubbard**
- Hamiltonian efektif van Vleck dan Floquet–Magnus sampai orde O(ω⁻²) atau lebih, diverifikasi
  terhadap ED: galat spektral vs ω harus menunjukkan slope sesuai orde pemotongan. Untuk drive
  harmonik tunggal, suku van Vleck O(ω⁻¹) lenyap (Bukov et al. 2015; Eckardt & Anisimovas 2015).
- Heating dan prethermalization: laju serapan energi vs ω, diuji terhadap skala exp(cω/J)
  (Abanin et al. 2015; Mori et al. 2016). Batas L yang bisa dicapai akan dilaporkan.
- Catatan desain (Q6): dengan J(t)=J₀cos ωt (rata-rata nol), H_eff orde terendah = H_int murni
  on-site, yang trivial. Pertimbangkan J(t)=J₀+J₁cos ωt atau drive on-site.

**D. Sirkuit kuantum dan NISQ**
- OTOC interferometrik dengan Trotter maju-mundur (melanjutkan P0-9), divalidasi terhadap OTOC eksak.
- Batas galat Trotter komutator (Childs et al., PRX 11, 011020, 2021) vs galat aktual.
- Sparse SYK (Xu, Susskind, Su, Swingle 2020): statistik spektral vs tingkat sparsity, plus jumlah
  gerbang dan kedalaman setelah transpile (Qiskit sebagai extra opsional).
- Membedakan scrambling dari dekoherensi: protokol teleportasi Yoshida–Yao (PRX 2019) atau
  normalisasi echo Loschmidt (Swingle & Yunger Halpern 2018, PERLU VERIFIKASI) di bawah model
  noise yang realistis dari P0-6.
- ZNE nyata: gate folding + Richardson (Mitiq; Giurgica-Tiron et al. 2020; Temme et al. 2017),
  divalidasi pada kasus dengan jawaban bebas-noise yang diketahui.

**E. QML (pertanyaan riset, bukan klaim)**
- Pertanyaan: apakah dinamika chaotic menyebabkan konsentrasi kernel eksponensial
  (Thanasilp et al., arXiv 2022)? Ukur Var[K] terhadap N dan t, untuk SYK, Hubbard t'=0, dan Hubbard t'≠0.
- Baseline klasik RBF-SVM dengan tuning hyperparameter (nested CV), ≥ 20 seed, uji berpasangan
  (Wilcoxon signed-rank atau corrected resampled t-test), ukuran efek, dan interval kepercayaan.
  Hasil negatif dilaporkan.

---

## 8. Target validasi dan kebijakan toleransi

| Besaran | Referensi | Toleransi | Fase |
|---|---|---|---|
| Identitas aljabar (CPTP, uniteritas, Tr_B, F(ρ,ρ), [H,simetri]) | Eksak | atol 1e-10 – 1e-12 | 0–1 |
| Peluruhan T1/T2 | exp(−t/T1), exp(−t/T2) | rtol 1e-10 | 0 |
| Konvergensi Trotter dan Floquet | Slope teoritis | ±0,15 pada slope log-log | 0 |
| OTOC sirkuit vs eksak | `compute_otoc_exact` | atol 1e-8 | 0 |
| ⟨r⟩ RMT tersampel | Atas et al. 2013 | 3σ + offset ukuran-hingga terukur | 1 |
| ⟨r⟩ SYK (GOE/GUE/GSE) | RMT dim-matched (kelas sesuai N mod 8) | 3σ statistik (bootstrap realisasi) | 1 |
| Hubbard U=0, t=0 | Fermion bebas, limit atomik | atol 1e-10 | 1 |
| Hubbard t'≠0 vs t'=0 | GOE vs ~Poisson | Dilaporkan dengan galat; **bukan** ambang lulus/gagal sampai literatur dicek | 1 |
| E₀ dan DOS SYK | García-García & Verbaarschot 2016 | Ditetapkan setelah nilai paper diverifikasi | 3B |
| SFF ramp | Form factor RMT tersampel | Ditetapkan di Fase 3B | 3B |

---

## 9. Referensi dan status verifikasi

Status: **B** = disebut di brief Anda; **I** = dari ingatan saya, cukup yakin; **V** = belum dicek,
PERLU VERIFIKASI. Semua referensi akan dicek ke sumber primer (DOI atau arXiv) sebelum dikutip
di kode atau README.

| Referensi | Dipakai untuk | Status |
|---|---|---|
| Atas, Bogomolny, Giraud, Roux, PRL 110, 084101 (2013) | ⟨r⟩ RMT | B, I |
| García-García & Verbaarschot, PRD 94, 126010 (2016) | Spektrum dan E₀ SYK | B, I (nomor halaman V) |
| Maldacena & Stanford, PRD 94, 106002 (2016) | Konvensi SYK | B, I |
| You, Ludwig, Xu, PRB 95, 115150 (2017) | Klasifikasi N mod 8 | V |
| Cotler et al., "Black holes and random matrices", JHEP (2017) | SFF | B, I (volume V) |
| Maldacena, Shenker, Stanford, JHEP (2016) | Batas chaos | I |
| Swingle, Bentsen, Schleier-Smith, Hayden, PRA 94, 040302 (2016) | OTOC interferometrik | I |
| Yoshida & Yao, PRX 9, 011006 (2019) | Scrambling vs dekoherensi | B, I |
| Childs, Su, Tran, Wiebe, Zhu, PRX 11, 011020 (2021) | Galat Trotter | B, I |
| Xu, Susskind, Su, Swingle, arXiv:2008.02303 (2020) | Sparse SYK | B, I |
| Thanasilp, Wang, Cerezo, Holmes, arXiv:2208.11060 (2022) | Konsentrasi kernel | B, I |
| Parker et al., PRX 9, 041017 (2019) | Krylov / Lanczos | I |
| Kobrin et al., PRL 126, 030602 (2021) | λ_L SYK pada N besar | V |
| Roberts, Stanford, Streicher, JHEP (2018) | Ukuran operator SYK | V |
| Bukov, D'Alessio, Polkovnikov, Adv. Phys. 64, 139 (2015); Eckardt & Anisimovas, NJP 17, 093039 (2015) | Ekspansi frekuensi tinggi | I |
| Abanin, De Roeck, Ho, Huveneers (2015); Mori, Kuwahara, Saito, PRL 116, 120401 (2016) | Prethermalization | I (detail V) |
| Sierant, Lewenstein, Zakrzewski, PRL 125, 156601 (2020) | POLFED | I |
| Bartsch & Gemmer, PRL 102, 110403 (2009) | Dynamical typicality | I |
| D'Alessio, Kafri, Polkovnikov, Rigol, Adv. Phys. 65, 239 (2016) | ETH | I |
| Page, PRL 71, 1291 (1993) | Kurva Page | I |
| Temme, Bravyi, Gambetta, PRL 119, 180509 (2017); Giurgica-Tiron et al. (2020); LaRose et al., Quantum 6, 774 (2022) | ZNE / Mitiq | I (detail V) |
| Swingle & Yunger Halpern, PRA (2018) | Echo Loschmidt dan resiliensi OTOC | V |

---

## 10. Keputusan (dicatat 2026-10-14)

Pemilik repo menjawab "setuju semua rekomendasi". Untuk butir yang tidak punya rekomendasi,
saya memakai pilihan default yang paling aman.

| # | Pertanyaan | Keputusan |
|---|---|---|
| Q1 | Struktur paket | **Pindah mekanis ke `src/quantum_chaos/` di branch pra-0** |
| Q2 | Versi Python | **CI 3.11–3.13 (+3.14 non-blocking), PennyLane terbaru (0.45.x)** |
| Q3 | Encoding kernel | **Hamiltonian encoding H(x)=H_model+Σxⱼ Oⱼ** |
| Q4 | Fungsi palsu | **Sesuai tabel P0-9** |
| Q5 | Alur git | *Tanpa rekomendasi, jadi tidak disetujui.* Default: hanya branch **lokal** yang ditumpuk (pra-0 → fase-0 → fase-1 → …). Tidak ada push atau PR tanpa izin eksplisit. |
| Q6 | Protokol drive Hubbard | *Masih terbuka*; diputuskan sebelum Fase 3C. Fase 0–1 mempertahankan J₀cos ωt. |
| Q7 | Identitas penulis | *Masih terbuka*; diperlukan sebelum `CITATION.cff` dibuat di Fase 2. |
| Q8 | Konfigurasi | **YAML + dataclass** |
| Q9 | Versi | **0.2.0** untuk rilis terkoreksi; **1.0.0** saat rilis Zenodo |
