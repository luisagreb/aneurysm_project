# Feature Metrics Documentation

This document describes all morphological features extracted by `analyze_structures.py`.

## Actin Features (Cytoskeleton)

| Feature | Unit | Description | Biological Significance |
|---------|------|-------------|------------------------|
| **Actin_Volume** | μm³ | Total volume of the actin cytoskeleton | Overall actin content; higher in cells with more developed stress fibers |
| **Actin_Skeleton_Length_Pixels** | pixels | Length of the skeletonized actin network | Network extent; longer = more branched/spread cytoskeleton |
| **Actin_Convex_Hull_Volume** | μm³ | Volume of the smallest convex shape enclosing the actin | Cell spread area in 3D; indicates how extended the cell is |
| **Actin_Solidity** | ratio (0-1) | Volume / Convex Hull Volume | Compactness; low solidity = more filamentous/branched structure |
| **Actin_Extent** | ratio (0-1) | Volume / Bounding Box Volume | How much of the bounding box is filled; low = irregular shape |
| **Actin_Fractional_Anisotropy** | ratio (0-1) | Directional organization from inertia tensor | High = aligned fibers (polarized cell); Low = random organization |
| **Actin_Major_Axis** | μm | Length of longest principal axis | Cell elongation direction |
| **Actin_Minor_Axis** | μm | Length of shortest principal axis | Cell width perpendicular to elongation |

---

## Mitochondria Features (Energy/Metabolism)

| Feature | Unit | Description | Biological Significance |
|---------|------|-------------|------------------------|
| **Mito_Volume** | μm³ | Total mitochondrial volume | Metabolic capacity; higher in metabolically active cells |
| **Mito_Surface_Area** | μm² | Total surface area of mitochondria | Related to cristae and ATP production capacity |
| **Mito_Sphericity** | ratio (0-1) | How close to a sphere: (π^⅓ × (6V)^⅔) / A | High = fragmented/rounded mitos; Low = elongated/tubular network |
| **Mito_Fragment_Count** | count | Number of discrete mitochondrial objects | High = fragmented (fission); Low = fused network |
| **Mito_Junction_Count** | count | Network branch points (degree > 2) | Network complexity; high = highly interconnected |
| **Mito_Branch_Count** | count | Number of skeleton branches | Network size and complexity |
| **Mito_Mean_Branch_Length** | μm | Average length per branch | Short = fragmented; Long = elongated tubules |
| **Mito_Total_Network_Length** | μm | Sum of all branch lengths | Total mitochondrial network extent |
| **Mito_Mean_Tortuosity** | ratio (≥1) | Branch length / Euclidean distance | Curvature; 1 = straight, higher = twisted |
| **Mito_Cyclomatic_Number** | count | E - N + C (graph complexity) | Number of independent loops in the network |

### Mitochondria Phenotype Interpretation:
- **Fission (fragmented)**: High Fragment_Count, High Sphericity, Low Branch_Count
- **Fusion (networked)**: Low Fragment_Count, Low Sphericity, High Junction_Count
- **Healthy**: Balanced fission/fusion dynamics, moderate network complexity
- **Stressed/TAA**: Often fragmented with reduced network connectivity

---

## Nucleus Features (Deformation/Mechanics)

| Feature | Unit | Description | Biological Significance |
|---------|------|-------------|------------------------|
| **Nucleus_Volume** | μm³ | Total nuclear volume | Cell size indicator; may change with ploidy or swelling |
| **Nucleus_Sphericity** | ratio (0-1) | How spherical the nucleus is | Low = deformed nucleus (mechanical stress or disease) |
| **Nucleus_Elongation** | ratio (≥1) | Major axis / Minor axis | 1 = round; Higher = stretched/elongated nucleus |
| **Nucleus_Flatness** | ratio (0-1) | Intermediate / Major axis | How "pancake-like" the nucleus is |
| **Nucleus_Solidity** | ratio (0-1) | Volume / Convex Hull Volume | Nuclear shape regularity; low = irregular/lobulated |

### Nuclear Shape Interpretation:
- **Healthy**: Spherical, regular shape (high sphericity, low elongation)
- **Mechanically stressed**: Elongated, deformed (high elongation, low sphericity)
- **Diseased (TAA)**: May show altered nuclear mechanics and deformation

---

## Feature Relationships

### Correlations to expect:
1. **Mito_Volume** ↔ **Mito_Surface_Area**: Strongly correlated (larger mitos have more surface)
2. **Mito_Fragment_Count** ↔ **Mito_Sphericity**: Positive correlation (fragments are rounder)
3. **Actin_Volume** ↔ **Actin_Convex_Hull_Volume**: Cell size correlation
4. **Nucleus_Sphericity** ↔ **Nucleus_Elongation**: Inverse correlation

### Key features for TAA classification:
Based on feature importance analysis:
1. **Mito_Sphericity** - Mitochondrial fragmentation state
2. **Mito_Fragment_Count** - Network fragmentation
3. **Actin_Solidity** - Cytoskeleton organization
4. **Nucleus_Sphericity** - Nuclear deformation

---

## Formulas

### Sphericity
```
Sphericity = (π^(1/3) × (6V)^(2/3)) / A
```
Where V = volume, A = surface area. Perfect sphere = 1.

### Solidity
```
Solidity = Volume / Convex_Hull_Volume
```
Ratio of actual volume to the smallest convex shape containing it.

### Cyclomatic Number (Graph Complexity)
```
C = E - N + P
```
Where E = edges (branches), N = nodes, P = connected components.

### Fractional Anisotropy
```
FA = sqrt(3/2) × sqrt(Σ(λᵢ - λ̄)² / Σλᵢ²)
```
Where λᵢ are eigenvalues of the inertia tensor. Range: 0 (isotropic) to 1 (anisotropic).
