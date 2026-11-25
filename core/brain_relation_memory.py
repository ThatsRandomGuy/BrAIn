"""Event-sourced relationship memory with explicit identity and evidence.

The grammar is deliberately bounded. Unrecognized statements remain searchable
source events; uncertainty and ambiguous references never become certain facts.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import os
import re
import tempfile
from brain_state_extraction import parse_number

VERSION=1
_ENTITY_PREDICATES={"located_in","owned_by","held_by","owned_items_stored_in"}
_SLOT_PREDICATES={"located_in","owned_by","held_by","owned_items_stored_in","cost","max_cost"}
_CONSTRAINT_PREDICATES={"requires","forbids","operates_from","independent_of","uses","cost","max_cost","required_location"}
_STOP=set("a an the and or but if then than as at by for from in into on of to with is are was were be been i me my we us our you your it its this that do does did can could should would will have has had now where which what who how".split())
_CODE=re.compile(r"^[a-z]*\d+[a-z0-9-]*$",re.I)
_QUALIFIED_CODE=re.compile(r"^(?:parcel|package|box|bin|case|crate|container|locker|bag)\s+([a-z]*\d+[a-z0-9-]*)$",re.I)

def norm(text):
    return re.sub(r"\s+"," ",str(text).strip().casefold()).strip(" .\"'")

def label(text):
    return re.sub(r"^(?:the|my|our|a|an)\s+","",str(text).strip(),flags=re.I).strip(" .\"'")

def words(text):
    return {w for w in re.findall(r"[a-z0-9]+",str(text).lower()) if w not in _STOP}

def _feature(text):
    value=norm(re.sub(r"^(?:using|use|from|on|a|an|the)\s+","",str(text).strip(),flags=re.I))
    if value.endswith(" dependency"): value=value[:-11].strip()
    if value.endswith("ies"): value=value[:-3]+"y"
    elif len(value)>4 and value.endswith("s") and not value.endswith("ss"): value=value[:-1]
    return value

@dataclass(frozen=True)
class SourceEvent:
    source_id:int
    text:str

@dataclass(frozen=True)
class Fact:
    subject:str
    predicate:str
    value:str
    target:str|None
    source_id:int
    support_ids:tuple[int,...]
    positive:bool=True
    certainty:str="explicit"
    order:int=0
    update:bool=False

class AmbiguousReference(ValueError):
    pass

class RelationMemory:
    def __init__(self,events=(),scope="conversation"):
        self.scope=str(scope)
        self.events={}
        for event in events:
            if isinstance(event,SourceEvent): sid,text=event.source_id,event.text
            else: sid,text=event["source_id"],event["text"]
            self._validate_event(sid,text)
            if sid in self.events and self.events[sid].text!=text:
                raise ValueError("A source ID cannot refer to different text")
            self.events[sid]=SourceEvent(sid,text)
        self._rebuild()

    @staticmethod
    def _validate_event(sid,text):
        if type(sid) is not int or sid<1 or not isinstance(text,str) or not text.strip():
            raise ValueError("Source IDs must be positive integers and source text nonempty")

    def add_event(self,source_id,text):
        self._validate_event(source_id,text)
        if source_id in self.events:
            if self.events[source_id].text!=text: raise ValueError("Conflicting source ID")
            return
        self.events[source_id]=SourceEvent(source_id,text)
        # Replaying makes out-of-order ingestion equivalent to chronological ingestion.
        self._rebuild()

    def _rebuild(self):
        self.aliases={}
        self.labels={}
        self.alias_proofs={}
        self.codes=defaultdict(set)
        self.facts=[]
        self.issues=[]
        self.eligible=set()
        self.ambiguous_labels=set()
        self.uncertain_slots={}
        self.uncertain_proofs={}
        for event in sorted(self.events.values(),key=lambda e:e.source_id):
            self._parse(event)

    def _issue(self,event,kind,detail,slots=()):
        issue={"source_id":event.source_id,"kind":kind,"detail":detail}
        if slots:issue["slots"]=[{"entity":eid,"relation":predicate} for eid,predicate in sorted(slots)]
        if issue not in self.issues:self.issues.append(issue)

    def entity(self,text,event):
        raw=str(text).strip()
        clean=label(re.sub(r"^another\s+","",raw,flags=re.I))
        key=norm(clean)
        if not key: raise AmbiguousReference("Empty entity")
        if key in {"it","this","that","they","them","he","she","him","her","other item"}:
            raise AmbiguousReference("Unresolved pronoun entity: "+clean)
        if key in self.ambiguous_labels:
            raise AmbiguousReference("Entity label is ambiguous: "+clean)
        if _CODE.fullmatch(key) and len(self.codes.get(key,set()))>1:
            raise AmbiguousReference("Identifier refers to multiple typed entities: "+clean)
        if re.match(r"^(?:a|an|another)\s+",raw,re.I) and key in self.aliases:
            eid=self.aliases[key]
            if any(f.source_id!=event.source_id and (f.subject==eid or f.target==eid) for f in self.facts):
                self.ambiguous_labels.add(key)
                raise AmbiguousReference("Repeated indefinite introduction: "+clean)
        if key in self.aliases:
            return self.aliases[key],self.alias_proofs.get(key,())
        if _CODE.fullmatch(key) and self.codes.get(key):
            matches=self.codes[key]
            if len(matches)!=1: raise AmbiguousReference("Identifier refers to multiple typed entities: "+clean)
            eid=next(iter(matches))
            return eid,tuple(sorted({sid for f in self.facts if f.subject==eid or f.target==eid for sid in f.support_ids}))
        eid=self.scope+"::"+key
        self.aliases[key]=eid
        self.labels[eid]=clean
        self.alias_proofs[key]=()
        qualified=_QUALIFIED_CODE.fullmatch(key)
        if qualified:self.codes[qualified.group(1)].add(eid)
        elif _CODE.fullmatch(key):self.codes[key].add(eid)
        return eid,()

    def _fact(self,subject,predicate,value,event,target=None,support=(),positive=True,certainty="explicit",update=False):
        proof=tuple(sorted(set(support)|{event.source_id}))
        if any(sid not in self.events for sid in proof):raise ValueError("Fact cites an unknown source")
        for existing in self.facts:
            if (existing.subject,existing.predicate,existing.value,existing.target,existing.source_id,existing.support_ids,existing.positive)==(subject,predicate,str(value),target,event.source_id,proof,bool(positive)):
                return existing
        fact=Fact(subject,predicate,str(value),target,event.source_id,proof,bool(positive),certainty,len(self.facts),bool(update))
        self.facts.append(fact)
        return fact

    def current(self,subject,predicate,at=None):
        relevant=[f for f in self.facts if f.subject==subject and f.predicate==predicate and (at is None or f.source_id<=at)]
        positives=[f for f in relevant if f.positive]
        if not positives:return None
        latest=max(f.source_id for f in positives)
        candidates=[f for f in positives if f.source_id==latest]
        values={f.value for f in candidates}
        chosen=candidates[-1]
        if len(values)!=1 and not chosen.update:return None
        uncertainty=self.uncertain_slots.get((subject,predicate),0)
        if uncertainty>=chosen.source_id and (at is None or uncertainty<=at):
            return None
        if any(not f.positive and f.value==chosen.value and
               (f.source_id>chosen.source_id or
                (f.source_id==chosen.source_id and (not chosen.update or f.order>chosen.order)))
               for f in relevant):
            return None
        return chosen

    def _rename(self,old,new,event):
        eid,proof=self.entity(old,event)
        new_label=label(new);key=norm(new_label)
        bound=self.aliases.get(key)
        if bound and bound!=eid:
            self._issue(event,"alias_collision",f"{new_label} already denotes another entity")
            self.ambiguous_labels.add(key)
            return
        self.aliases[key]=eid
        self.alias_proofs[key]=tuple(sorted(set(proof)|{event.source_id}))
        self._fact(eid,"alias",new_label,event,support=proof)

    def _transfer(self,giver,item,receiver,event):
        actor,actor_proof=self.entity(giver,event)
        clean=norm(label(item))
        references={"it","this","that","item","received item","that received item","same item"}
        proof=set(actor_proof)
        if clean in references:
            candidates=[]
            for eid in self.labels:
                held=self.current(eid,"held_by")
                if held and held.target==actor:candidates.append((eid,held))
            if len(candidates)!=1:
                ambiguity_proof=tuple(sorted({event.source_id}|{sid for _,old in candidates for sid in old.support_ids}))
                for candidate,_ in candidates:
                    self.uncertain_slots[(candidate,"held_by")]=event.source_id
                    self.uncertain_proofs[(candidate,"held_by")]=ambiguity_proof
                self._issue(event,"ambiguous_reference",f"{giver}: {item} has {len(candidates)} known referents")
                return
            obj,previous=candidates[0]
            proof.update(previous.support_ids)
            certainty="resolved_reference"
        else:
            obj,obj_proof=self.entity(item,event);proof.update(obj_proof)
            certainty="explicit"
        holder,holder_proof=self.entity(receiver,event);proof.update(holder_proof)
        self._fact(obj,"held_by",self.labels[holder],event,target=holder,support=proof,certainty=certainty,update=True)

    @staticmethod
    def _nonactual_change(clause):
        if re.search(r"\b(will|would|should|could|might|must|tomorrow|next week|next month|if|unless|plan(?:ning)? to|intend(?:ing)? to|going to|want to|did not|didn't|do not|don't|have not|haven't|has not|hasn't|never|not yet)\b",clause,re.I):return True
        return bool(re.match(r"^(?:please\s+)?(?:put|place|move|relocate|set|stash|tuck|insert|store|remove)\b",clause,re.I))

    def _invalidate_unresolved_change(self,event,clause):
        """Conservatively retire affected state when a reported change was not parsed."""
        change=re.search(r"\b(moved|relocated|put|placed|set|stashed|tucked|slid|slipped|inserted|deposited|positioned|repositioned|shifted|carted|carried|transported|removed|took|taken|brought|packed|dropped|squirrelled|squirreled|renamed|transferred|changed|swapped|replaced|handed|gave|passed)\b",clause,re.I)
        if not change:return
        # Plans and negated reports do not establish that an old fact has changed.
        if self._nonactual_change(clause):return
        predicate="held_by" if change.group(1).lower() in {"handed","gave","passed","transferred"} else "located_in"
        mentioned={eid for alias,eid in self.aliases.items()
                   if re.search(r"(?<!\w)"+re.escape(alias)+r"(?!\w)",norm(clause))}
        if not mentioned and re.search(r"\b(it|them|that item|this item)\b",clause,re.I):
            mentioned={f.subject for f in self.facts if f.predicate==predicate}
        slots={(eid,predicate) for eid in mentioned
               if any(f.subject==eid and f.predicate==predicate for f in self.facts)}
        for eid,pred in slots:
            self.uncertain_slots[(eid,pred)]=event.source_id
            self.uncertain_proofs[(eid,pred)]=(event.source_id,)
        self._issue(event,"unsupported_state_change","A reported change was not resolved; affected state is unknown",slots)

    def _issue_is_unresolved(self,issue):
        slots=issue.get("slots",[])
        return not slots or any((self.current(slot["entity"],slot["relation"]) is None or
            self.current(slot["entity"],slot["relation"]).source_id<=issue["source_id"]) for slot in slots)

    def _parse(self,event):
        text=event.text.strip()
        if "?" in text or re.match(r"^(where|what|which|who|why|how|is|are|can|could|do|does)\b",text,re.I):
            return
        self.eligible.add(event.source_id)
        if re.search(r"\b(maybe|perhaps|possibly|probably|might|may be|could be|I think)\b",text,re.I):
            self._issue(event,"uncertain_source","Tentative statements are not promoted to certain facts")
            return
        for sentence in re.split(r";|\.(?!\d)",text):
            clause=sentence.strip()
            if not clause:continue
            explicit_update=bool(re.match(r"^(?:actually|correction|update)\b",clause,re.I))
            clause=re.sub(r"^(?:actually|correction|update)\b\s*[:,]?\s*","",clause,flags=re.I)
            before=len(self.facts)
            try:
                if self._nonactual_change(clause) and re.search(r"\b(moved|relocated|put|placed|set|stashed|tucked|slid|slipped|inserted|stored|kept|deposited|positioned|shifted|carted|carried|removed|handed|gave|passed)\b",clause,re.I):
                    continue
                movement=re.fullmatch(r"(?:I|we)\s+(?:have\s+)?(?:moved|relocated)\s+(.+?)(?:\s+from\s+(.+?))?\s+to\s+(.+)",clause,re.I)
                if not movement:
                    movement=re.fullmatch(r"(.+?)\s+(?:(?:was|has been)\s+)?(?:moved|relocated)\s+(?:from\s+(.+?)\s+)?to\s+(.+)",clause,re.I)
                if movement:
                    subject,origin,destination=movement.groups()
                    destination=re.split(r"\s+(?:instead of|rather than)\b",destination,flags=re.I)[0].strip()
                    if re.match(r"^(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|yesterday|next\s+week)\b",label(destination),re.I):
                        self._issue(event,"unsupported_temporal_move","A reschedule is not a physical location");continue
                    a,ap=self.entity(subject,event)
                    if origin:
                        old,op=self.entity(origin,event)
                        self._fact(a,"located_in",self.labels[old],event,target=old,support=ap+op)
                    b,bp=self.entity(destination,event)
                    self._fact(a,"located_in",self.labels[b],event,target=b,support=ap+bp,update=True)
                    continue
                placement=re.fullmatch(r"(?:I|we|[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)\s+(?:(?:have|has)\s+)?(?:put|placed|set|stashed|tucked|slid|slipped|inserted|deposited|positioned|repositioned)\s+(.+?)\s+(?:in|into|inside|on|onto|under|beside|at)\s+(.+)",clause)
                if not placement:
                    placement=re.fullmatch(r"(.+?)\s+(?:was|has been)\s+(?:put|placed|set|stashed|tucked|slid|slipped|inserted|deposited|positioned|repositioned)\s+(?:in|into|inside|on|onto|under|beside|at)\s+(.+)",clause,re.I)
                if placement:
                    item,destination=placement.groups()
                    a,ap=self.entity(item,event);b,bp=self.entity(destination,event)
                    self._fact(a,"located_in",self.labels[b],event,target=b,support=ap+bp,update=True)
                    continue
                stored=re.fullmatch(r"(.+?)\s+(?:keep|keeps|kept|store|stores|stored)\s+(.+?)\s+(?:in|inside|on)\s+(.+)",clause,re.I)
                if stored:
                    _,item,destination=stored.groups()
                    a,ap=self.entity(item,event);b,bp=self.entity(destination,event)
                    self._fact(a,"located_in",self.labels[b],event,target=b,support=ap+bp)
                    continue
                renamed=re.fullmatch(r"(?:I|we)\s+(?:have\s+)?renamed\s+(.+)",clause,re.I)
                if renamed:
                    body=renamed.group(1)
                    explicit=re.fullmatch(r"(.+?)\s+to\s+(.+)",body,re.I)
                    if explicit:
                        self._rename(*explicit.groups(),event)
                    else:
                        tail=label(body)
                        candidates=[]
                        for known in self.aliases:
                            prefix=re.match(re.escape(known)+r"\b",tail,re.I)
                            if prefix and tail[prefix.end():].strip():
                                candidates.append((prefix.end(),known,tail[prefix.end():].strip()))
                        if candidates:
                            _,old,new=max(candidates)
                            self._rename(old,new,event)
                        else:
                            self._issue(event,"ambiguous_rename","Use an explicit 'to' or a known old entity name")
                    continue
                match=re.fullmatch(r"(.+?)\s+is\s+(?:now\s+)?(?:called|named)\s+(.+)",clause,re.I)
                if match:
                    self._rename(*match.groups(),event);continue
                match=re.fullmatch(r"(.+?)\s+(?:handed|gave|passed)\s+(.+?)\s+to\s+(.+)",clause,re.I)
                if match:
                    self._transfer(*match.groups(),event);continue
                match=re.fullmatch(r"(?:everything\s+(.+?)\s+owns|all\s+of\s+(.+?)'s\s+belongings)\s+(?:is|are)\s+(?:kept|stored)\s+(?:in|inside|on)\s+(.+)",clause,re.I)
                if match:
                    owner=match.group(1) or match.group(2)
                    a,ap=self.entity(owner,event);b,bp=self.entity(match.group(3),event)
                    self._fact(a,"owned_items_stored_in",self.labels[b],event,target=b,support=ap+bp);continue
                match=re.fullmatch(r"(.+?)\s+(?:(does not|no longer)\s+belongs?|belongs?)\s+to\s+(.+)",clause,re.I)
                if match:
                    a,ap=self.entity(match.group(1),event);b,bp=self.entity(match.group(3),event)
                    self._fact(a,"owned_by",self.labels[b],event,target=b,support=ap+bp,positive=not bool(match.group(2)));continue
                match=re.fullmatch(r"(.+?)\s+(?:kept|holds|has retained)\s+(.+)",clause,re.I)
                if match:
                    owner,op=self.entity(match.group(1),event);item,ip=self.entity(match.group(2),event)
                    self._fact(item,"held_by",self.labels[owner],event,target=owner,support=op+ip);continue
                match=re.fullmatch(r"(.+?)\s+(?:is|are)\s+(?:(now|currently|still)\s+)?(?:(not)\s+)?(?:(?:packed|stored|kept|located)\s+)?(?:in|inside|on|under|beside|at)\s+(.+)",clause,re.I)
                if not match:
                    simple=re.fullmatch(r"(.+?)\s+(?:rests|sits)\s+(?:on|in|inside)\s+(.+)",clause,re.I)
                    if simple:
                        groups=(simple.group(1),None,None,simple.group(2))
                    else:groups=None
                else:groups=match.groups()
                if groups:
                    subject,_,negative,destination=groups
                    destination=re.split(r"\s+(?:instead of|rather than)\b",destination,flags=re.I)[0].strip()
                    if re.search(r"\b(?:is|are)\b",destination):
                        self._issue(event,"unsupported_compound","Compound location assertion needs separate clauses");continue
                    a,ap=self.entity(subject,event);b,bp=self.entity(destination,event)
                    self._fact(a,"located_in",self.labels[b],event,target=b,support=ap+bp,positive=not bool(negative),update=explicit_update or groups[1]=="now");continue
                # Inherit a subject only across explicit parallel predicates.
                parts=re.split(r"\s+and\s+(?=(?:must|needs|requires|has|works|crosses|costs|avoids)\b)",clause,flags=re.I)
                actor=None
                for part in parts:
                    if actor is not None:part=actor+" "+part
                    match=re.match(r"^(.+?)\s+((?:no longer\s+)?requires|must|should|needs|has|works|crosses|costs|avoids|does)\b",part,re.I)
                    if not match:continue
                    actor=match.group(1)
                    eid,ep=self.entity(actor,event)
                    body=part[match.start(2):]
                    cost=re.fullmatch(r"costs\s+(.+?)\s+(dollars?|euros?|pounds?)",body,re.I)
                    bound=re.fullmatch(r"must\s+spend\s+(no more than|at most|under|less than|below)\s+(.+?)\s+(dollars?|euros?|pounds?)",body,re.I)
                    if cost or bound:
                        numeric=parse_number((cost.group(1) if cost else bound.group(2)).lower())
                        if numeric is None:
                            self._issue(event,"unsupported_number","Quantity was not parsed exactly");continue
                        unit=(cost.group(2) if cost else bound.group(3)).lower().rstrip("s")
                        value=numeric+" "+unit
                        if bound:
                            value=("<=" if bound.group(1).lower() in {"no more than","at most"} else "<")+" "+value
                        self._fact(eid,"cost" if cost else "max_cost",value,event,support=ep);continue
                    predicate=None;value=None;positive=True
                    patterns=[
                        ("required_location",r"must\s+(?:stay|remain|be kept)\s+(?:in|inside|on)\s+(.+)"),
                        ("required_location",r"should\s+(?:stay|remain|be kept)\s+(?:in|inside|on)\s+(.+)"),
                        ("forbids",r"must\s+not\s+use\s+(.+)"),
                        ("forbids",r"must\s+travel\s+without\s+(?:using\s+)?(.+)"),
                        ("requires",r"must\s+use\s+(.+)"),
                        ("independent_of",r"(?:needs|requires)\s+no\s+(.+)"),
                        ("requires",r"(?:needs|requires)\s+(.+)"),
                        ("requires",r"does\s+not\s+require\s+(.+)"),
                        ("requires",r"no\s+longer\s+requires\s+(.+)"),
                        ("operates_from",r"works\s+from\s+(.+)"),
                        ("independent_of",r"has\s+no\s+(.+?)\s+dependency"),
                        ("uses",r"crosses\s+no\s+(.+)"),
                        ("uses",r"avoids\s+(.+)"),
                        ("uses",r"crosses\s+(.+)")]
                    for kind,pattern in patterns:
                        feature=re.fullmatch(pattern,body,re.I)
                        if feature:
                            predicate=kind;value=label(feature.group(1)) if kind=="required_location" else _feature(label(feature.group(1)))
                            positive=not (body.lower().startswith(("does not require","no longer requires","crosses no","avoids")))
                            break
                    if predicate and value:
                        self._fact(eid,predicate,value,event,support=ep,positive=positive)
                if len(self.facts)==before:
                    self._invalidate_unresolved_change(event,clause)
            except AmbiguousReference as exc:
                self._issue(event,"ambiguous_entity",str(exc))
                self._invalidate_unresolved_change(event,clause)

    def _latest_facts(self):
        latest=[]
        for eid in self.labels:
            for predicate in _SLOT_PREDICATES:
                current=self.current(eid,predicate)
                if current:latest.append(current)
        # Set-like feature facts are versioned independently, by subject/predicate/value.
        groups=defaultdict(list)
        for fact in self.facts:
            if fact.predicate not in _SLOT_PREDICATES and fact.predicate!="alias":
                groups[(fact.subject,fact.predicate,fact.value)].append(fact)
        for facts in groups.values():
            sid=max(f.source_id for f in facts)
            current=[f for f in facts if f.source_id==sid]
            if len({f.positive for f in current})==1:latest.append(current[-1])
        return sorted(set(latest),key=lambda f:(f.source_id,f.subject,f.predicate,f.value))

    def query(self,question,max_hops=3):
        normalized=norm(question)
        if not normalized:return {"active":False}
        matches=[(alias,eid) for alias,eid in self.aliases.items()
                 if re.search(r"(?<!\w)"+re.escape(alias)+r"(?!\w)",normalized)]
        # Prefer full entity names over aliases which are merely contained in them.
        seeds={eid for alias,eid in matches if not any(alias!=other and alias in other for other,_ in matches)}
        query_issues=[{"kind":"ambiguous_query_entity","detail":alias}
                      for alias,_ in matches if alias in self.ambiguous_labels]
        for code in re.findall(r"\b[a-z]*\d+[a-z0-9-]*\b",normalized):
            candidates=self.codes.get(code,set())
            seeds.update(candidates)
            if len(candidates)>1:
                query_issues.append({"kind":"ambiguous_query_entity","detail":code+" names multiple entities"})
        lexical=[]
        query_words=words(question)
        for sid in self.eligible:
            overlap=len(query_words & words(self.events[sid].text))
            if overlap:lexical.append((overlap,sid))
        lexical.sort(reverse=True)
        if not seeds:
            selected_ids={sid for _,sid in lexical[:3]}
            seeds={f.subject for f in self.facts if f.source_id in selected_ids}
        if re.search(r"\b(compatible|satisfies|restrictions?|requirements?|rules|constraints?)\b",normalized):
            seeds.update(f.subject for f in self.facts if f.predicate in _CONSTRAINT_PREDICATES)
        if not seeds:
            proof={sid for _,sid in lexical[:3]}
            if re.search(r"\b(it|they|them|this item|that item)\b",normalized):
                return {"active":True,"evidence_bundle_version":1,"sources":[],
                    "current_facts":[],"aliases":[],"supported_answer":"","answer_status":"unresolved",
                    "coverage":"unresolved query reference","facts_are_partial":True,
                    "policy":"The queried referent is not identified. Ask for clarification; do not guess.",
                    "issues":[{"kind":"unresolved_query_reference","detail":"Name the entity explicitly"}]}
            issues=[issue for issue in self.issues if issue["source_id"] in proof and self._issue_is_unresolved(issue)]
            if issues:
                return {"active":True,"evidence_bundle_version":1,
                    "sources":[{"id":sid,"text":self.events[sid].text} for sid in sorted(proof)],
                    "current_facts":[],"aliases":[],"supported_answer":"",
                    "answer_status":"unresolved","facts_are_partial":True,
                    "coverage":"uncertain source evidence",
                    "policy":"No certain fact was established. Do not promote tentative statements to current state.",
                    "issues":issues}
            return {"active":False}
        facts=self._latest_facts()
        nodes=set(seeds);selected=set()
        for _ in range(max_hops):
            found=[f for f in facts if f.subject in nodes or (f.target is not None and f.target in nodes)]
            selected.update(found)
            expanded={f.target for f in found if f.target is not None}|{f.subject for f in found}
            if expanded<=nodes:break
            nodes.update(expanded)
        proof={sid for fact in selected for sid in fact.support_ids}
        unresolved=[]
        for eid in sorted(nodes):
            for predicate in _SLOT_PREDICATES:
                history=[f for f in self.facts if f.subject==eid and f.predicate==predicate]
                if history and self.current(eid,predicate) is None:
                    proof.update(sid for f in history for sid in f.support_ids)
                    proof.update(self.uncertain_proofs.get((eid,predicate),()))
                    unresolved.append({"kind":"unresolved_state","detail":self.labels[eid]+" / "+predicate})
        # Include explicit alias bindings even when they were used by query names.
        aliases=[]
        for alias,eid in sorted(self.aliases.items()):
            if eid in nodes and self.alias_proofs.get(alias):
                refs=self.alias_proofs[alias];proof.update(refs)
                aliases.append({"name":alias,"entity":self.labels[eid],"sources":list(refs)})
        # Raw matching statements remain evidence even when the bounded grammar did not parse them.
        proof.update(sid for _,sid in lexical[:3])
        for issue in self.issues:
            if not self._issue_is_unresolved(issue):continue
            source_text=norm(self.events[issue["source_id"]].text)
            if any(eid in nodes and re.search(r"(?<!\w)"+re.escape(alias)+r"(?!\w)",source_text)
                   for alias,eid in self.aliases.items()):
                proof.add(issue["source_id"])
        related_issues=query_issues+unresolved+[issue for issue in self.issues if issue["source_id"] in proof and self._issue_is_unresolved(issue)]
        source_rows=[{"id":sid,"text":self.events[sid].text} for sid in sorted(proof)]
        rows=[{"subject":self.labels[f.subject],"relation":f.predicate,
               "value":f.value,"positive":f.positive,"sources":list(f.support_ids)}
              for f in sorted(selected,key=lambda f:(f.source_id,f.subject,f.predicate,f.value))]
        supported=""
        # Only direct, proof-supported identity/state lookups get a deterministic answer.
        target=next(iter(seeds)) if len(seeds)==1 else None
        if target and not related_issues:
            if re.search(r"\b(who|whose)\b",normalized):
                predicate="owned_by" if re.search(r"\b(owns?|owner|whose|belongs?)\b",normalized) else "held_by"
                held=self.current(target,predicate)
                if held:supported=held.value
            elif re.search(r"\bwhere\b",normalized):
                path=[];visited=set();current=target;cycle=False
                while current not in visited:
                    visited.add(current)
                    loc=self.current(current,"located_in")
                    if not loc and current==target:
                        owner=self.current(current,"owned_by")
                        if owner and owner.target:
                            storage=self.current(owner.target,"owned_items_stored_in")
                            if storage:
                                proof.update(owner.support_ids);proof.update(storage.support_ids)
                                if storage.target==target:
                                    related_issues.append({"kind":"location_cycle","detail":"Ownership storage would place the item inside itself"})
                                else:
                                    supported=self.labels[target]+" is in "+storage.value+"."
                        if not supported:
                            related_issues.append({"kind":"missing_item_location","detail":"An owner's location is not evidence of an item's location"})
                        break
                    if not loc:break
                    path.append(loc);proof.update(loc.support_ids)
                    current=loc.target
                    if current is None:break
                else:cycle=True
                if cycle:
                    related_issues.append({"kind":"location_cycle","detail":"No terminal location is established"})
                elif path:
                    if re.search(r"\bcontainer\b.*\b(holding|containing)\b",normalized) and len(path)<2:
                        related_issues.append({"kind":"missing_container_location","detail":"The container's location is not recorded"})
                    elif re.search(r"\b(original|initial|first)\b",normalized):
                        history=[f for f in self.facts if f.subject==target and f.predicate=="located_in" and f.positive]
                        if history:
                            oldest=min(history,key=lambda f:f.source_id)
                            proof.update(oldest.support_ids)
                            supported=self.labels[target]+" was originally in "+oldest.value+"."
                    else:
                        supported="; ".join(self.labels[f.subject]+" is in "+f.value for f in path)+"."
        source_rows=[{"id":sid,"text":self.events[sid].text} for sid in sorted(proof)]
        if not rows and not related_issues:return {"active":False}
        return {"active":True,"evidence_bundle_version":1,
            "sources":source_rows,"current_facts":rows if not related_issues else [],
            "last_confirmed_facts":rows if related_issues else [],"aliases":aliases,
            "supported_answer":supported,
            "answer_status":"supported" if supported else "requires_interpretation" if not related_issues else "unresolved",
            "coverage":"retrieved subset of recorded events",
            "facts_are_partial":True,
            "policy":"Sources are authoritative. Facts index the sources; they are not an exhaustive summary. Unknown or conflicted properties stay unknown.",
            "issues":related_issues}

    def snapshot(self,source_id):
        return RelationMemory([e for e in self.events.values() if e.source_id<=source_id],self.scope)

    def save(self,path):
        path=Path(path)
        payload={"version":VERSION,"scope":self.scope,
                 "events":[asdict(e) for e in sorted(self.events.values(),key=lambda e:e.source_id)]}
        checksum=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        document={**payload,"checksum":checksum}
        path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,delete=False) as file:
            temporary=Path(file.name);json.dump(document,file)
        try:os.replace(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()

    @classmethod
    def load(cls,path):
        document=json.loads(Path(path).read_text(encoding="utf-8"))
        checksum=document.pop("checksum",None)
        if set(document)!={"version","scope","events"} or type(document.get("version")) is not int or document.get("version")!=VERSION:
            raise ValueError("Unsupported relation-memory version or fields")
        if not isinstance(document["events"],list) or not isinstance(document["scope"],str) or not document["scope"]:
            raise ValueError("Invalid relation-memory document")
        if hashlib.sha256(json.dumps(document,sort_keys=True).encode()).hexdigest()!=checksum:
            raise ValueError("Relation-memory checksum mismatch")
        return cls(document["events"],document["scope"])

def fit_atomic_bundle(view,count,budget):
    """A proof bundle is admitted whole or rejected without exposing partial facts."""
    if count(view)<=budget:return view,{"atomic_bundle":True,"bundle_rejected":False}
    rejected={"evidence_status":"unavailable_under_budget",
              "reason":"A complete relationship evidence bundle does not fit; answer UNKNOWN rather than guess."}
    if count(rejected)>budget:raise ValueError("Instructions and question leave no room for a safe rejection")
    return rejected,{"atomic_bundle":True,"bundle_rejected":True,
                     "required_source_ids":[s["id"] for s in view.get("sources",[])]}


def _assertion_events(records):
    events=[]
    for record in records:
        if record.route_type in {"memory_question","factual_question","planning_request",
                                  "goal_request","transform_request","hybrid_request"}:
            continue
        events.append(SourceEvent(record.turn_id,record.raw_user_prompt))
    return events


def has_relevant_relation_history(records,question):
    """Require a recorded entity/subject anchor, not generic lexical overlap."""
    events=_assertion_events(records)
    if not events:return False
    try:
        index=RelationMemory(events)
        context=index.query(question)
    except ValueError:
        return False
    if not context.get("active"):return False
    normalized=norm(question).rstrip("?!.")
    if re.match(r"^(?:define\b|what is (?:a|an)\b|how (?:do|does)\b|why\b)",normalized) or "in general" in normalized:
        return False
    if any(alias not in _STOP and re.search(r"(?<!\w)"+re.escape(alias)+r"(?!\w)",normalized)
           for alias in index.aliases):return True
    # Tentative assertions can have a subject without creating a certain entity.
    for event in events:
        for clause in re.split(r";|\.(?!\d)",event.text):
            subject=re.match(r"\s*(.+?)\s+(?:is|are|was|were|might|may|must|belongs?|needs?|requires?)\b",clause,re.I)
            if subject:
                anchor=norm(label(subject.group(1)))
                if len(words(anchor))>=2 and re.search(r"(?<!\w)"+re.escape(anchor)+r"(?!\w)",normalized):return True
    return bool(re.fullmatch(r"(?:where (?:is|are)|who (?:has|holds)) (?:it|this|that|them)(?: now)?",normalized))


def context_from_turns(records,question,scope="conversation",constraint_extractor=None):
    """Index user assertions only. Generated answers are never treated as evidence."""
    events=_assertion_events(records)
    try:
        result=RelationMemory(events,scope).query(question)
        if result.get("active") and constraint_extractor is not None and re.search(r"\b(constraints?|requirements?)\b",question,re.I):
            # Reuse trusted, raw-text extraction for scalar constraints outside the
            # relation grammar. Never trust model answers or stored response metadata.
            latest={}
            for event in sorted(events,key=lambda item:item.source_id):
                if "?" in event.text:continue
                for observation in constraint_extractor(event.text):
                    if observation.get("state_type")=="planning_constraint":
                        latest[observation["state_key"]]=(observation["display_text"],event.source_id)
            if latest:
                existing={source["id"]:source for source in result["sources"]}
                source_map={event.source_id:event for event in events}
                for _,sid in latest.values():
                    existing[sid]={"id":sid,"text":source_map[sid].text}
                result["sources"]=[existing[sid] for sid in sorted(existing)]
                result["recorded_constraints"]=[{"text":text,"source":sid} for text,sid in latest.values()]
                if re.match(r"\s*what\b",question,re.I) and not re.search(r"\b(compatible|satisfies|choose|best|which option)\b",question,re.I):
                    result["supported_answer"]="Your recorded constraints: "+"; ".join(dict.fromkeys(text for text,_ in latest.values()))+"."
                    result["answer_status"]="supported"
        return result
    except ValueError as exc:
        return {"active":True,"evidence_bundle_version":1,"sources":[],
                "current_facts":[],"last_confirmed_facts":[],"aliases":[],
                "supported_answer":"","answer_status":"unresolved",
                "coverage":"invalid source history","facts_are_partial":True,
                "policy":"Source history could not be validated. Do not assert an answer.",
                "issues":[{"kind":"invalid_source_history","detail":str(exc)}]}

def apply_to_legacy_state(state,context):
    """Retire conflicting legacy entity state without discarding unrelated constraint data."""
    merged=dict(state)
    managed={norm(f["subject"]) for f in context.get("current_facts",[])+context.get("last_confirmed_facts",[])}
    managed.update(norm(a["name"]) for a in context.get("aliases",[]))
    managed.update(norm(a["entity"]) for a in context.get("aliases",[]))
    items=[]
    for item in state.get("items",[]):
        # Legacy scalar constraints remain available for legacy consumers. They are
        # not relabeled as facts verified by this index.
        if item.get("state_type") in {"planning_constraint","dietary_constraint"}:
            items.append({**item,"provenance":"legacy_extractor"})
    for fact in context.get("current_facts",[]):
        subject,predicate,value=fact["subject"],fact["relation"],fact["value"]
        if predicate=="located_in" and fact["positive"]:
            kind="object_location";key="object_location:"+norm(subject).replace(" ","_")
            display=subject+" is in "+value
        elif predicate in {"held_by","owned_by"} and fact["positive"]:
            kind="entity_holder" if predicate=="held_by" else "entity_owner"
            key=kind+":"+norm(subject).replace(" ","_")
            display=subject+" "+predicate+" "+value
        elif predicate in _CONSTRAINT_PREDICATES:
            kind="planning_constraint"
            key="relation:"+predicate+":"+norm(subject)+":"+norm(value)
            display=(subject+" in "+value if predicate=="required_location" else
                     subject+" "+("" if fact["positive"] else "not ")+predicate+" "+value)
        else:continue
        items.append({"state_key":key,"state_type":kind,"entity":subject,"current_value":value,
                      "display_text":display,"source_ids":fact["sources"],"provenance":"event_relations"})
    merged["items"]=items
    merged["item_count"]=len(items)
    merged["relation_memory"]=context
    return merged
