import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 加载模型和分词器
# model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
# model_name = "/home/yandong/Documents/um-data/models/Orac-mini-3B"
model_name = "/home/yandong/Documents/um-data/models/MPT-7B-Chat"
# model_name = "/home/yandong/Documents/um-data/models/InternLM2-chat-7B"
# model_name = "/home/yandong/Documents/um-data/models/Vicuna-7B"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts = ['What sits on top of the Main Building at Notre Dame?',
           'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
           'What is in front of the Notre Dame Main Building?',
           'The Basilica of the Sacred heart at Notre Dame is beside to which structure?',
           'What is the Grotto at Notre Dame?',
           'When did the Scholastic Magazine of Notre dame begin publishing?',
           "How often is Notre Dame's the Juggler published?"]
input_text = '''University of Chicago / Facebook Amid a high-profile controversy at the University of Chicago over the value of trigger warnings and "intellectual safe spaces," a male student is suing the Hyde Park school on the grounds that it is not a safe space for young men. The unnamed student has been accused of sexual assault by his peers at least twice, but the university found he was not at fault after investigating the claims through its own process for resolving on-campus disputes between students. The lawsuit, filed last week, demands $175,000 in damages against the university. The university has come under scrutiny in recent years for its handling of sexual assault accusations among students and is being investigated by the federal government for violating Title IX. Now this student says the school has swung too far in the other direction, creating a "gender-biased, hostile environment against males, like John Doe, based in part on [the University of Chicago's] pattern and practice of investigating and disciplining male students who accept physical contact initiated by female students, retaliating against male students, and providing female students preferential treatment under its Title IX policies." The lawsuit, which can be read in full here, details the relationships John Doe had with two unnamed women referred to as Jane Doe and Jane Roe in 2013. The lawsuit claims the relationships and his sexual activities with them were always consensual, but Jane Doe began publicly accusing him of sexual assaulting them in 2016, several years after the fact. The lawsuit excerpts a series of Tumblr blog posts written by one of his accusers in 2013 and 2014 as evidence that the relationship was in fact consensual. Among other issues, the male student says he was harassed by the female students after one Tweeted about the alleged assault and a group of students staged a protest to boycott a student theater production he directed, and after his named was placed on "the Hyde Park List,"—a very unofficial list of University of Chicago students accused of sexual misconduct that was disseminated anonymously in 2014. The complaint relies on some common arguments used to refute sexual assault claims women bring against men: that the women are scorned lovers trying to re-write history out of anger, and that people who believe sexual assault allegations are biased against men because of the mainstream cultural image of men as more interested in casual sex than women. The lawsuit claims one female student had a "vendetta" against him and was retaliating after being rejected, and that the university operates under “archaic assumptions that female students do not sexually assault or harass their fellow male students because females are less sexually promiscuous than men.” [H/T Jezebel] A University of Chicago male student from New York accused twice of sexual assault slammed the school's conduct in a new lawsuit, charging that "UC routinely portrays a large portion of their male students as sexual predators." The student, named as John Doe in the lawsuit, claims the university violated Title IX with its unfair treatment. He was subjected to a "fundamentally unfair, arbitrary and capricious disciplinary procedure that violates both Title IX and UC's policies and/or procedures related to allegations of sexual misconduct," according to the lawsuit obtained by the Chicago Maroon, a student newspaper at the school. Title IX is in place to ensure that universities don't discriminate against students and faculty based on gender. The male student, from Somers, N.Y., charges that two women, named as Jane Doe and Jane Roe, falsely accused him of sexual assault. Jane Doe is also being sued, according to the Maroon. What you should know about reporting sex assault on campus The suit alleges that the school refused to allow John Doe to file a Title IX complaint against Jane Doe for harassment and retaliation, yet UC allowed Jane Doe to file a Title IX complaint to retaliate for an alleged sexual assault two years earlier. John Doe and Jane Doe became acquainted before their first semester in September 2013 through a Facebook page for accepted students, according to the lawsuit. John Doe and Roe met during their first semester. The lawsuit states that after John Doe broke off his relationships with the two women, Roe filed a complaint with the school falsely accusing John Doe of sexual misconduct in the spring of 2014. The accusation was regarding alleged conduct that took place the previous December. The school rejected Roe's claims "despite UC's anti-male gender bias conduct and gross negligence in conducting an investigation," according to the lawsuit. Brown student could be allowed back after sex assault suspension However, Roe then launched a vendetta against the male student, according to the lawsuit. She placed his name on the so-called "Hyde Park List," a Tumblr page accusing six current and former male students of sexual assault or sexual harassment. Roe also "falsely and maliciously advised members of UC's community that John Doe was a sexual predator," according to the suit. The school did not take any steps to "correct" Roe's conduct, according to the suit, as John Doe was told the school's "confidentiality" policies prevented him from personally refuting Roe's defamation within "UC's Community." Roe then blogged in October of 2014 that the school was forcing her to participate in class "with the person who sexually assaulted [HER]." The school ordered John Doe removed from the physics lab they were both in, despite his strong opposition, the lawsuit states. UC investigating sexually explicit ‘gag reflex' sign near campus In addition, the school "unlawfully alleged a right to discipline" the male student because his sister responded to Roe's "fallacious" posts on Twitter. The other woman mentioned in the suit, Jane Doe, first mentioned in her blog that she had been "sexually assaulted" in November of 2014, according to the complaint. However, in her own "vernacular," according to the lawsuit, the male student and Jane Doe "hooked up" in September of 2013 but did not have sexual intercourse. Jane Doe harassed the male student in a series of tweets earlier this year, according to the lawsuit. The male student filed a Title IX complaint against her, but the school took no action. The suit seeks $175,000 in damages, according to the Maroon. A male student twice investigated for sexual assault by the University of Chicago has sued the school, saying its handling of the complaints against him demonstrated a “gender-based, hostile environment against males.” The student, identified in the lawsuit as “John Doe” from Somers, N.Y., claims the University failed to treat him fairly after a University disciplinary panel cleared him of a sexual assault charge in 2014. The suit also accuses a female student of defamation for publicly saying he assaulted her and others. The complaint was filed on August 24 in federal court in Chicago. The suit asks for more than $175,000 in damages. The case against the University states that, in its treatment of the plaintiff in the case, “UC was motivated by pro-female, anti-male bias that was, at least in part, adopted to refute criticism within the student body and public press that UC was turning a blind eye to female complaints of sexual assault.” The suit is part of a recent trend of litigation by male students at American universities who claim to have been discriminated against in sexual assault investigations in violation of Title IX, a federal statute that prohibits sex discrimination at colleges and universities that receive federal funding. So far, these claims have generally failed, though last month a federal appeals court ruled in favor of a similar suit against Columbia University. The suit involves two separate allegations of sexual assault against “John Doe” filed with the University as Title IX complaints. The first allegation, filed by a female student called “Jane Roe” in the lawsuit, was decided in “John Doe’s” favor in 2014. The second was filed this year by another female student, called “Jane Doe” in the lawsuit, over an alleged instance of sexual assault in 2013. “Jane Doe” is being sued, along with the University. The student the suit refers to as “Jane Roe” said “John Doe” assaulted her in the spring of 2014. A College disciplinary hearing in May of that year “found that the preponderance of the evidence did not support [“Jane Roe’s”] allegation that [“John Doe”] sexually assaulted her,” according to a letter sent to the plaintiff by former assistant dean of students Kathleen Ford. The letter was included as an exhibit in the suit. The suit alleges that “Jane Roe” told other people at the University of Chicago that “John Doe” had raped her and that she added his name to the “Hyde Park List,” a Tumblr page that listed students it said had committed sexual assault or harassment. “Roe” is not a defendant in the lawsuit. Instead, the suit points to the University’s decision to remove the plaintiff from a physics lab with “Roe” after the case had been decided, and its failure to stop her from repeating her allegations, as examples of the University’s bias against him as a male student. In May of this year, the plaintiff directed a TAPS-sponsored play called The Bald Soprano, according to an email released as an exhibit in the case. Before the play’s debut, flyers were posted on bulletin boards around campus urging people to boycott the show, and a protest was staged near the theater on the play’s opening night. “For me, it’s about supporting survivors, boycotting events put on by perpetrators of sexual assault, and accountability in RSOs,” then second-year Emma Maltby, who participated in the protest, said when interviewed by The Maroon at the time. At the protest, students held signs with slogans like “Support survivors, not perpetrators,” “Boycott rapists,” “Silence = violence,” and “Hold your friends accountable.” Some protesters also had tape over their mouths, with “Support survivors” written on the tape. Students protest in Reynolds Club.'''
inputs = tokenizer(input_text, return_tensors="pt").to(device)


# response = model.generate(inputs['input_ids'], max_length=100)
# print(tokenizer.decode(response[0]))


# 计算每一层的输入输出相似度
def compute_cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    a = a.view(a.size(0), -1)  # 展平
    b = b.view(b.size(0), -1)  # 展平
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    similarity = torch.mm(a_norm, b_norm.t())
    return similarity.item()


# 获取模型的每一层的输入输出
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    # outputs = model.generate(inputs['input_ids'], max_length=20)
    # outputs = model.generate(inputs['input_ids'], max_length=20, return_dict_in_generate=True, output_hidden_states=True)

    # GenerateDecoderOnlyOutput(sequences=tensor([[1, 1128, 4049, 338, 24337, 360, 420, 29915, 29879, 278,
    #                                              12028, 29887, 1358, 6369, 29973, 13, 3664, 276, 360, 420]]),
    #                           scores=None, logits=None, attentions=None, hidden_states=

    hidden_states = outputs.hidden_states  # 包含每一层的输入输出
    # print(hidden_states[0].shape)
    # print(hidden_states[1].shape)
    # print(hidden_states[32].shape)
    # print("output:", outputs.keys())


# 遍历每一层
for i in range(1, len(hidden_states) - 1):
    input_layer = hidden_states[i]
    output_layer = hidden_states[i + 1]
    # print("layer", i, " ", input_layer, output_layer)
    similarity = compute_cosine_similarity(input_layer, output_layer)
    print(similarity)
