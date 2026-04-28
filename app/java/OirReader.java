import loci.formats.ImageReader;
import loci.formats.FormatTools;
import loci.formats.meta.IMetadata;
import loci.formats.services.OMEXMLServiceImpl;
import ome.units.UNITS;
import java.io.FileOutputStream;
import java.io.DataOutputStream;
import java.io.PrintStream;

public class OirReader {
    public static void main(String[] args) throws Exception {
        String inFile = args[0];
        String outFile = args[1];

        ImageReader reader = new ImageReader();
        IMetadata meta = new OMEXMLServiceImpl().createOMEXMLMetadata();
        reader.setMetadataStore(meta);
        reader.setId(inFile);

        int sizeX = reader.getSizeX(), sizeY = reader.getSizeY();
        int sizeZ = reader.getSizeZ(), sizeC = reader.getSizeC();
        int bpp = FormatTools.getBytesPerPixel(reader.getPixelType());

        // Physical pixel sizes (default 1.0 if unavailable)
        double pz = 1.0, py = 1.0, px = 1.0;
        try { pz = meta.getPixelsPhysicalSizeZ(0).value(UNITS.MICROMETER).doubleValue(); } catch (Exception e) {}
        try { py = meta.getPixelsPhysicalSizeY(0).value(UNITS.MICROMETER).doubleValue(); } catch (Exception e) {}
        try { px = meta.getPixelsPhysicalSizeX(0).value(UNITS.MICROMETER).doubleValue(); } catch (Exception e) {}

        DataOutputStream dos = new DataOutputStream(new FileOutputStream(outFile));
        // Header: C Z Y X bpp pz py px (doubles)
        dos.writeInt(sizeC); dos.writeInt(sizeZ); dos.writeInt(sizeY); dos.writeInt(sizeX); dos.writeInt(bpp);
        dos.writeDouble(pz); dos.writeDouble(py); dos.writeDouble(px);

        for (int c = 0; c < sizeC; c++)
            for (int z = 0; z < sizeZ; z++) {
                int idx = reader.getIndex(z, c, 0);
                dos.write(reader.openBytes(idx));
            }

        dos.close();
        reader.close();

        // Print summary to stdout for Python to parse
        System.out.println("OK " + sizeC + " " + sizeZ + " " + sizeY + " " + sizeX + " " + pz + " " + py + " " + px);
    }
}
