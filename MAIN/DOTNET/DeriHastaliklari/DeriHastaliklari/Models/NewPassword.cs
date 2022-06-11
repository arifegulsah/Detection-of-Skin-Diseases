using System.ComponentModel.DataAnnotations;

namespace DeriHastaliklari.Models
{
    public class NewPassword
    {
        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string NewPass { get; set; }

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string NewPassRepeat { get; set; }


        public int PatientId { get; set; }
    }
}
