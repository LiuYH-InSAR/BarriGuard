from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from night_lamp.report_render import render_lamp_report,health_style,GREEN,RED,GRAY


class ReportColors(unittest.TestCase):
    def test_only_decided_normal_is_green_and_decided_failure_red(self):
        self.assertEqual(health_style('normal')[0],GREEN)
        for state in ['not_flashing','missing','lost']:
            self.assertEqual(health_style(state)[0],RED)
        for state in ['unobservable','insufficient_observation','no_flash','below_threshold','nested_identity_ambiguous',None,'unexpected']:
            self.assertEqual(health_style(state)[0],GRAY,state)

    def test_rectangles_follow_report_status_not_counts_or_last_detection(self):
        frame=np.full((200,400,3),40,np.uint8);original=frame.copy()
        lamps=[dict(lamp_id='normal123456',status='normal',box=[20,40,50,70]),
            dict(lamp_id='failed654321',status='not_flashing',box=[80,40,110,70]),
            dict(lamp_id='unknownABCDEF',status='insufficient_observation',box=[140,40,170,70],pulses=999)]
        canvas,metadata=render_lamp_report(frame,lamps,'cam','2026-10-31T07:00:00+08:00')
        for x,color in [(20,GREEN),(80,RED),(140,GRAY)]:np.testing.assert_array_equal(canvas[55,x],color)
        np.testing.assert_array_equal(frame,original)
        self.assertEqual(canvas.shape,(276,400,3))
        self.assertEqual(metadata['counts'],dict(normal=1,failed=1,unknown=1))
        self.assertEqual(metadata['source_frame_header_added_px'],0)

    def test_short_ids_remain_distinct_when_suffixes_collide(self):
        lamps=[dict(lamp_id=identity,status='normal',box=[20,40,50,70]) for identity in ['L1111abcdef','L2222abcdef']]
        _,metadata=render_lamp_report(np.zeros((100,200,3),np.uint8),lamps,'cam','07:00')
        self.assertEqual(len(set(metadata['display_ids'].values())),2)

    def test_no_lamps_does_not_invent_green_or_red_records(self):
        _,metadata=render_lamp_report(np.zeros((100,200,3),np.uint8),[],'cam','07:00')
        self.assertEqual(metadata['counts'],dict(normal=0,failed=0,unknown=0))


if __name__=='__main__':unittest.main()
