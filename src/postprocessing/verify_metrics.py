import numpy as np
import unittest
from skimage import draw
import sys
import os

# Add directory to path to import analyze_structures
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyze_structures import analyze_mito, analyze_actin, analyze_nucleus, calculate_sphericity

class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        self.voxel_size = (1.0, 1.0, 1.0)
        self.shape = (100, 100, 100)
        
    def create_sphere(self, radius, center=(50, 50, 50)):
        mask = np.zeros(self.shape, dtype=np.uint8)
        z, y, x = np.ogrid[:self.shape[0], :self.shape[1], :self.shape[2]]
        dist_sq = (x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2
        mask[dist_sq <= radius**2] = 1
        return mask
        
    def create_rod(self, length, radius=2):
        # Create a rod along Z axis
        mask = np.zeros(self.shape, dtype=np.uint8)
        rr, cc = draw.disk((50, 50), radius)
        # simplistic rod
        mask[20:20+length, rr, cc] = 1
        return mask

    def test_mito_sphere(self):
        """Test Mitochondria metrics on a perfect sphere."""
        radius = 10
        mask = self.create_sphere(radius)
        
        metrics = analyze_mito(mask, self.voxel_size)
        
        # Expected Volume: 4/3 * pi * r^3
        expected_vol = (4/3) * np.pi * radius**3
        # Allow some error due to voxelization
        self.assertAlmostEqual(metrics['Volume'], expected_vol, delta=0.1 * expected_vol)
        
        # Expected Sphericity for sphere is 1.0
        # Voxelized sphere is not perfect, but should be high > 0.9
        self.assertGreater(metrics['Sphericity'], 0.85)
        
        # Fragment count should be 1
        self.assertEqual(metrics['Fragment_Count'], 1)
        
    def test_mito_fragments(self):
        """Test Fragment Count with two disjoint spheres."""
        mask = self.create_sphere(5, center=(30, 30, 30))
        mask += self.create_sphere(5, center=(70, 70, 70))
        
        metrics = analyze_mito(mask, self.voxel_size)
        self.assertEqual(metrics['Fragment_Count'], 2)
        
    def test_actin_length(self):
        """Test Actin Skeleton Length on a rod."""
        length = 40
        mask = self.create_rod(length, radius=3)
        
        metrics = analyze_actin(mask, self.voxel_size)
        
        # Skeleton length should be approximately the length of the rod
        # Skeletonize might not be perfect endpoints, allow some delta
        self.assertAlmostEqual(metrics['Skeleton_Length_Pixels'], length, delta=5)
        
    def test_nucleus_elongation(self):
        """Test Nucleus Elongation on an ellipsoid."""
        mask = np.zeros(self.shape, dtype=np.uint8)
        # Ellipsoid radii: 20, 10, 10
        # Equation: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
        z, y, x = np.ogrid[:100, :100, :100]
        mask[((x-50)/10)**2 + ((y-50)/10)**2 + ((z-50)/20)**2 <= 1] = 1
        
        metrics = analyze_nucleus(mask, self.voxel_size)
        
        # Major axis should be Z (~40 diameter), Minor axes ~20
        # Elongation ~ 2.0
        self.assertGreater(metrics['Elongation'], 1.5)
        self.assertLess(metrics['Elongation'], 2.5)

if __name__ == '__main__':
    unittest.main()
