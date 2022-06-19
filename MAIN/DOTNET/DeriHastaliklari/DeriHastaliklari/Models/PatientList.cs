namespace DeriHastaliklari.Models
{
    public class PatientList
    {
        public string TcNo { get; set; }
        public string Name { get; set; }
        public string Surname { get; set; }
        public bool Medicine { get; set; }
        public string MedicineText { get; set; }
        public bool Disease { get; set; }
        public string DiseaseText { get; set; }
        public bool Allergy { get; set; }
        public string AllergyText { get; set; }
        public string Itch { get; set; }
        public string Pain { get; set; }
        public string Cont { get; set; }
        public string Additional { get; set; }
    }
}
