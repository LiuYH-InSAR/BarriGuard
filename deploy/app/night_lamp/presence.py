"""Conservative physical absence, independent of night flash health."""
import math


class PresenceLedger:
    def __init__(self,monitor):self.monitor=monitor

    def pause(self,lamp=None):
        for item in ([lamp] if lamp is not None else self.monitor.lamps.values()):
            item.presence['previous_valid']=False

    def positive(self,lamp,stamp,source='day_body'):
        p=lamp.presence
        if p.get('status')=='missing':
            p.setdefault('events',[]).append(dict(at=stamp,event='missing_recovered',source=source))
        p.update(status='present',reference=lamp.reference,last_present_at=stamp,
                 previous_valid=False,history=[],bucket=None,valid_s=0.,checks=0)
        p['events']=p.get('events',[])[-32:]

    def observe(self,lamp,stamp,source_ts,verdict,details=None):
        cfg=self.monitor.params;p=lamp.presence
        if verdict=='present':self.positive(lamp,stamp);return
        if p.get('reference')!=lamp.reference:
            p.update(reference=lamp.reference,history=[],bucket=None,status='unknown',previous_valid=False)
        p.update(last_checked_at=stamp,last_evidence=dict(details or {},verdict=verdict))
        bucket=int(stamp//cfg.cadence_s)*cfg.cadence_s
        if p.get('bucket')!=bucket:
            p.update(bucket=bucket,valid_s=0.,checks=0,previous_valid=False)
        previous=p.get('previous_source_ts');last_wall=p.get('previous_wall')
        valid=(verdict=='absent' and previous is not None and last_wall is not None and p.get('previous_valid')
               and 0<source_ts-previous<=cfg.presence_period_s*1.5
               and 0<stamp-last_wall<=cfg.presence_period_s*1.5)
        if valid:p['valid_s']+=min(source_ts-previous,stamp-last_wall)
        p.update(previous_source_ts=source_ts,previous_wall=stamp,previous_valid=verdict=='absent')
        if verdict!='absent':return
        p['checks']+=1;p['last_absence_at']=stamp
        if p['valid_s']<cfg.effective_s or p['checks']<cfg.missing_min_checks:return
        rows={r['bucket']:r for r in p.get('history',[])}
        rows[bucket]=dict(bucket=bucket,reference=lamp.reference,valid_s=p['valid_s'],
                          checks=p['checks'],ended=stamp,evidence=dict(details or {}))
        p['history']=sorted(rows.values(),key=lambda r:r['bucket'])[-96:]
        expected=[bucket-i*cfg.cadence_s for i in range(cfg.missing_windows)]
        if all(b in rows and rows[b]['reference']==lamp.reference and rows[b]['valid_s']>=cfg.effective_s
               and rows[b]['checks']>=cfg.missing_min_checks for b in expected):
            if p.get('status')!='missing':
                p.setdefault('events',[]).append(dict(at=stamp,event='missing_confirmed'))
            p.update(status='missing',confirmed_at=stamp)
        elif p.get('status')!='missing':p['status']='suspected_missing'
        p['events']=p.get('events',[])[-32:]

    def status(self,lamp,stamp):
        p=lamp.presence;cfg=self.monitor.params
        if (p.get('status')!='missing' or p.get('reference')!=lamp.reference
                or not math.isfinite(p.get('last_absence_at',float('nan')))
                or not 0<=stamp-p['last_absence_at']<=cfg.missing_max_age_s):return None
        rows={r['bucket']:r for r in p.get('history',[])};last=p.get('bucket')
        if last is None and rows:last=max(rows)
        if last is None:return None
        expected=[last-i*cfg.cadence_s for i in range(cfg.missing_windows)]
        if not all(b in rows and rows[b]['reference']==lamp.reference
            and rows[b]['valid_s']>=cfg.effective_s and rows[b]['checks']>=cfg.missing_min_checks for b in expected):return None
        return 'missing'

    def migrate(self,lamp):
        p=lamp.presence
        if p.get('history'):p['prior_view_history']=p['history'][-self.monitor.params.missing_windows:]
        p.update(reference=lamp.reference,status='unknown',history=[],bucket=None,
                 valid_s=0.,checks=0,previous_valid=False)
