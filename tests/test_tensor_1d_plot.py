import math,unittest
from benchmarks.tensor_1d.plot import aggregate_rows


def row(psnr,input_bytes=400,compressed_bytes=50,compression_time=1):
    return {'psnr':psnr,'slice_num':12,'input_bytes':input_bytes,'compressed_bytes':compressed_bytes,'compression_time':compression_time,'decompression_time':2,'compression_preprocessing_time':0,'decompression_preprocessing_time':0,'valid':True,'max_error_over_eb':1}


class AggregationTests(unittest.TestCase):
    def test_equal_fields_and_infinite_psnr(self):
        result=aggregate_rows([row(20),row(math.inf,compressed_bytes=25,compression_time=3)],'sz3',1e-3)
        self.assertEqual(result['br'],3)
        self.assertAlmostEqual(result['nmse'],.005)
        self.assertAlmostEqual(result['normalized_psnr'],-10*math.log10(.005))
        self.assertEqual(result['compression_MB_s'],800/1e6/4)

    def test_unequal_fields_use_point_weights(self):
        result=aggregate_rows([row(20),row(30,input_bytes=1200)],'sz3',1e-3)
        self.assertAlmostEqual(result['nmse'],(.01+3*.001)/4)

    def test_all_exact_and_invalid_status(self):
        failed=row(math.inf)
        failed['valid']=False
        failed['max_error_over_eb']=2
        result=aggregate_rows([row(math.inf),failed],'zfp',1e-3)
        self.assertEqual(result['normalized_psnr'],math.inf)
        self.assertEqual(result['invalid_field_num'],1)
        self.assertFalse(result['valid'])


if __name__=='__main__':
    unittest.main()
